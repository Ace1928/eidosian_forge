import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import utils
from keras.src.applications import imagenet_utils
from keras.src.engine import sequential
from keras.src.engine import training as training_lib
from tensorflow.python.util.tf_export import keras_export
class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return x / keep_prob * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'drop_path_rate': self.drop_path_rate})
        return config