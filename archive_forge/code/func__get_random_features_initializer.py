import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.engine import base_layer
from keras.src.engine import input_spec
from tensorflow.python.util.tf_export import keras_export
def _get_random_features_initializer(initializer, shape):
    """Returns Initializer object for random features."""

    def _get_cauchy_samples(loc, scale, shape):
        probs = np.random.uniform(low=0.0, high=1.0, size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))
    random_features_initializer = initializer
    if isinstance(initializer, str):
        if initializer.lower() == 'gaussian':
            random_features_initializer = initializers.RandomNormal(stddev=1.0)
        elif initializer.lower() == 'laplacian':
            random_features_initializer = initializers.Constant(_get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))
        else:
            raise ValueError(f'Unsupported `kernel_initializer`: "{initializer}" Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}')
    return random_features_initializer