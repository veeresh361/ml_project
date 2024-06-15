import pytest
import joblib
import numpy as np
import os

@pytest.fixture(scope='module')
def setup_model():
    model_path = os.path.join('models', 'svc_model.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    test_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    return model, scaler, test_data

def test_prediction(setup_model):
    model, scaler, test_data = setup_model
    scaled_data = scaler.transform(test_data)
    prediction = model.predict(scaled_data)
    assert prediction[0] in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


if __name__ == '__main__':
    pytest.main()