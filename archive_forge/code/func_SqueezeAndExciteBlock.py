import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def SqueezeAndExciteBlock(filters_in, se_filters, name=None):
    """Implements the Squeeze & Excite block (https://arxiv.org/abs/1709.01507).

    Args:
      filters_in: input filters to the block
      se_filters: filters to squeeze to
      name: name prefix

    Returns:
      A function object
    """
    if name is None:
        name = str(backend.get_uid('squeeze_and_excite'))

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(name=name + '_squeeze_and_excite_gap', keepdims=True)(inputs)
        x = layers.Conv2D(se_filters, (1, 1), activation='relu', kernel_initializer='he_normal', name=name + '_squeeze_and_excite_squeeze')(x)
        x = layers.Conv2D(filters_in, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name=name + '_squeeze_and_excite_excite')(x)
        x = tf.math.multiply(x, inputs)
        return x
    return apply