import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _check_factor_range(self, input_number):
    if input_number > 1.0 or input_number < -1.0:
        raise ValueError(self._FACTOR_VALIDATION_ERROR + f'Received: input_number={input_number}')