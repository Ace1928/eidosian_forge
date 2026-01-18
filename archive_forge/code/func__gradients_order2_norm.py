import copy
import tensorflow.compat.v2 as tf
from keras.src.engine import data_adapter
from keras.src.layers import deserialize as deserialize_layer
from keras.src.models import Model
from keras.src.saving.object_registration import register_keras_serializable
from keras.src.saving.serialization_lib import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export
def _gradients_order2_norm(self, gradients):
    norm = tf.norm(tf.stack([tf.norm(grad) for grad in gradients if grad is not None]))
    return norm