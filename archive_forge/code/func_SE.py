import sys
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def SE(in_filters: int, se_ratio: float=0.25, expand_ratio: int=1, name=None):
    """Squeeze and Excitation block."""
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if name is None:
        counter = backend.get_uid('se_')
        name = f'se_{counter}'

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(inputs)
        if bn_axis == 1:
            se_shape = (x.shape[-1], 1, 1)
        else:
            se_shape = (1, 1, x.shape[-1])
        x = layers.Reshape(se_shape, name=name + '_se_reshape')(x)
        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))
        x = layers.Conv2D(filters=num_reduced_filters, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same', use_bias=True, activation='relu', name=name + '_se_reduce')(x)
        x = layers.Conv2D(filters=4 * in_filters * expand_ratio, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same', use_bias=True, activation='sigmoid', name=name + '_se_expand')(x)
        return layers.multiply([inputs, x], name=name + '_se_excite')
    return apply