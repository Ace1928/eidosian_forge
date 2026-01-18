import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _adjust_block(p, ip, filters, block_id=None):
    """Adjusts the input `previous path` to match the shape of the `input`.

    Used in situations where the output number of filters needs to be changed.

    Args:
        p: Input tensor which needs to be modified
        ip: Input tensor whose shape needs to be matched
        filters: Number of output filters to be matched
        block_id: String block_id

    Returns:
        Adjusted Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == 'channels_first' else -1
    img_dim = 2 if backend.image_data_format() == 'channels_first' else -2
    ip_shape = backend.int_shape(ip)
    if p is not None:
        p_shape = backend.int_shape(p)
    with backend.name_scope('adjust_block'):
        if p is None:
            p = ip
        elif p_shape[img_dim] != ip_shape[img_dim]:
            with backend.name_scope(f'adjust_reduction_block_{block_id}'):
                p = layers.Activation('relu', name=f'adjust_relu_1_{block_id}')(p)
                p1 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name=f'adjust_avg_pool_1_{block_id}')(p)
                p1 = layers.Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, name=f'adjust_conv_1_{block_id}', kernel_initializer='he_normal')(p1)
                p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = layers.AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name=f'adjust_avg_pool_2_{block_id}')(p2)
                p2 = layers.Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, name=f'adjust_conv_2_{block_id}', kernel_initializer='he_normal')(p2)
                p = layers.concatenate([p1, p2], axis=channel_dim)
                p = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'adjust_bn_{block_id}')(p)
        elif p_shape[channel_dim] != filters:
            with backend.name_scope(f'adjust_projection_block_{block_id}'):
                p = layers.Activation('relu')(p)
                p = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f'adjust_conv_projection_{block_id}', use_bias=False, kernel_initializer='he_normal')(p)
                p = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'adjust_bn_{block_id}')(p)
    return p