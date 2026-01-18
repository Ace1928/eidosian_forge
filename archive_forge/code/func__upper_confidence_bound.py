import numpy as np
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
def _upper_confidence_bound(x):
    x = x.reshape(1, -1)
    mu, sigma = self.gpr.predict(x, return_std=True)
    return mu - self.beta * sigma