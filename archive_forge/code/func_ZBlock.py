import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def ZBlock(filters_in, filters_out, group_width, stride=1, squeeze_excite_ratio=0.25, bottleneck_ratio=0.25, name=None):
    """Implementation of Z block Reference: [Fast and Accurate Model
    Scaling](https://arxiv.org/abs/2103.06877).

    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      bottleneck_ratio: inverted bottleneck ratio
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid('zblock'))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(f'Input filters({filters_in}) and output filters({filters_out})are not equal for stride {stride}. Input and output filters must be equal for stride={stride}.')
        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)
        inv_btlneck_filters = int(filters_out / bottleneck_ratio)
        x = layers.Conv2D(inv_btlneck_filters, (1, 1), use_bias=False, kernel_initializer='he_normal', name=name + '_conv_1x1_1')(inputs)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_1x1_1_bn')(x)
        x = tf.nn.silu(x)
        x = layers.Conv2D(inv_btlneck_filters, (3, 3), use_bias=False, strides=stride, groups=groups, padding='same', kernel_initializer='he_normal', name=name + '_conv_3x3')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_3x3_bn')(x)
        x = tf.nn.silu(x)
        x = SqueezeAndExciteBlock(inv_btlneck_filters, se_filters, name=name)
        x = layers.Conv2D(filters_out, (1, 1), use_bias=False, kernel_initializer='he_normal', name=name + '_conv_1x1_2')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_1x1_2_bn')(x)
        if stride != 1:
            return x
        else:
            return x + inputs
    return apply