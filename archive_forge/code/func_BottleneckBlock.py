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
def BottleneckBlock(filters: int, strides: int, use_projection: bool, bn_momentum: float=0.0, bn_epsilon: float=1e-05, activation: str='relu', se_ratio: float=0.25, survival_probability: float=0.8, name=None):
    """Bottleneck block variant for residual networks with BN."""
    if name is None:
        counter = backend.get_uid('block_0_')
        name = f'block_0_{counter}'

    def apply(inputs):
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        shortcut = inputs
        if use_projection:
            filters_out = filters * 4
            if strides == 2:
                shortcut = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name=name + '_projection_pooling')(inputs)
                shortcut = Conv2DFixedPadding(filters=filters_out, kernel_size=1, strides=1, name=name + '_projection_conv')(shortcut)
            else:
                shortcut = Conv2DFixedPadding(filters=filters_out, kernel_size=1, strides=strides, name=name + '_projection_conv')(inputs)
            shortcut = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_projection_batch_norm')(shortcut)
        x = Conv2DFixedPadding(filters=filters, kernel_size=1, strides=1, name=name + '_conv_1')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + 'batch_norm_1')(x)
        x = layers.Activation(activation, name=name + '_act_1')(x)
        x = Conv2DFixedPadding(filters=filters, kernel_size=3, strides=strides, name=name + '_conv_2')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_batch_norm_2')(x)
        x = layers.Activation(activation, name=name + '_act_2')(x)
        x = Conv2DFixedPadding(filters=filters * 4, kernel_size=1, strides=1, name=name + '_conv_3')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_batch_norm_3')(x)
        if 0 < se_ratio < 1:
            x = SE(filters, se_ratio=se_ratio, name=name + '_se')(x)
        if survival_probability:
            x = layers.Dropout(survival_probability, noise_shape=(None, 1, 1, 1), name=name + '_drop')(x)
        x = layers.Add()([x, shortcut])
        return layers.Activation(activation, name=name + '_output_act')(x)
    return apply