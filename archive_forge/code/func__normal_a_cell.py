import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _normal_a_cell(ip, p, filters, block_id=None):
    """Adds a Normal cell for NASNet-A (Fig. 4 in the paper).

    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id

    Returns:
        A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == 'channels_first' else -1
    with backend.name_scope(f'normal_A_block_{block_id}'):
        p = _adjust_block(p, ip, filters, block_id)
        h = layers.Activation('relu')(ip)
        h = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f'normal_conv_1_{block_id}', use_bias=False, kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'normal_bn_1_{block_id}')(h)
        with backend.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), block_id=f'normal_left1_{block_id}')
            x1_2 = _separable_conv_block(p, filters, block_id=f'normal_right1_{block_id}')
            x1 = layers.add([x1_1, x1_2], name=f'normal_add_1_{block_id}')
        with backend.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), block_id=f'normal_left2_{block_id}')
            x2_2 = _separable_conv_block(p, filters, (3, 3), block_id=f'normal_right2_{block_id}')
            x2 = layers.add([x2_1, x2_2], name=f'normal_add_2_{block_id}')
        with backend.name_scope('block_3'):
            x3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f'normal_left3_{block_id}')(h)
            x3 = layers.add([x3, p], name=f'normal_add_3_{block_id}')
        with backend.name_scope('block_4'):
            x4_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f'normal_left4_{block_id}')(p)
            x4_2 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f'normal_right4_{block_id}')(p)
            x4 = layers.add([x4_1, x4_2], name=f'normal_add_4_{block_id}')
        with backend.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, block_id=f'normal_left5_{block_id}')
            x5 = layers.add([x5, h], name=f'normal_add_5_{block_id}')
        x = layers.concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name=f'normal_concat_{block_id}')
    return (x, ip)