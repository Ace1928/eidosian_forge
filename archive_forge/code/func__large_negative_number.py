from keras.src import activations
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == 'float16':
        return -30000.0
    return -1000000000.0