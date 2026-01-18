import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def MBConvBlock(input_filters: int, output_filters: int, expand_ratio=1, kernel_size=3, strides=1, se_ratio=0.0, bn_momentum=0.9, activation='swish', survival_probability: float=0.8, name=None):
    """MBConv block: Mobile Inverted Residual Bottleneck."""
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if name is None:
        name = backend.get_uid('block0')

    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same', data_format='channels_last', use_bias=False, name=name + 'expand_conv')(inputs)
            x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, name=name + 'expand_bn')(x)
            x = layers.Activation(activation, name=name + 'expand_activation')(x)
        else:
            x = inputs
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depthwise_initializer=CONV_KERNEL_INITIALIZER, padding='same', data_format='channels_last', use_bias=False, name=name + 'dwconv2')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, name=name + 'bn')(x)
        x = layers.Activation(activation, name=name + 'activation')(x)
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
            se = layers.Conv2D(filters_se, 1, padding='same', activation=activation, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_reduce')(se)
            se = layers.Conv2D(filters, 1, padding='same', activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + 'se_expand')(se)
            x = layers.multiply([x, se], name=name + 'se_excite')
        x = layers.Conv2D(filters=output_filters, kernel_size=1, strides=1, kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same', data_format='channels_last', use_bias=False, name=name + 'project_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, name=name + 'project_bn')(x)
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                x = layers.Dropout(survival_probability, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
            x = layers.add([x, inputs], name=name + 'add')
        return x
    return apply