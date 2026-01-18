import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def XBlock(filters_in, filters_out, group_width, stride=1, name=None):
    """Implementation of X Block.

    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid('xblock'))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(f'Input filters({filters_in}) and output filters({filters_out}) are not equal for stride {stride}. Input and output filters must be equal for stride={stride}.')
        groups = filters_out // group_width
        if stride != 1:
            skip = layers.Conv2D(filters_out, (1, 1), strides=stride, use_bias=False, kernel_initializer='he_normal', name=name + '_skip_1x1')(inputs)
            skip = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_skip_bn')(skip)
        else:
            skip = inputs
        x = layers.Conv2D(filters_out, (1, 1), use_bias=False, kernel_initializer='he_normal', name=name + '_conv_1x1_1')(inputs)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_1x1_1_bn')(x)
        x = layers.ReLU(name=name + '_conv_1x1_1_relu')(x)
        x = layers.Conv2D(filters_out, (3, 3), use_bias=False, strides=stride, groups=groups, padding='same', kernel_initializer='he_normal', name=name + '_conv_3x3')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_3x3_bn')(x)
        x = layers.ReLU(name=name + '_conv_3x3_relu')(x)
        x = layers.Conv2D(filters_out, (1, 1), use_bias=False, kernel_initializer='he_normal', name=name + '_conv_1x1_2')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_conv_1x1_2_bn')(x)
        x = layers.ReLU(name=name + '_exit_relu')(x + skip)
        return x
    return apply