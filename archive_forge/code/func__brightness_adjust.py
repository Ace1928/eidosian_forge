import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def _brightness_adjust(self, images):
    rank = images.shape.rank
    if rank == 3:
        rgb_delta_shape = (1, 1, 1)
    elif rank == 4:
        rgb_delta_shape = [tf.shape(images)[0], 1, 1, 1]
    else:
        raise ValueError(f'Expected the input image to be rank 3 or 4. Got inputs.shape = {images.shape}')
    rgb_delta = self._random_generator.random_uniform(shape=rgb_delta_shape, minval=self._factor[0], maxval=self._factor[1])
    rgb_delta = rgb_delta * (self._value_range[1] - self._value_range[0])
    rgb_delta = tf.cast(rgb_delta, images.dtype)
    images += rgb_delta
    return tf.clip_by_value(images, self._value_range[0], self._value_range[1])