import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _expand_tensor_with_local_replica_group(inputs):
    """Reshape the input tensor to have an extra dimension of replica group.

    Under the DTensor usage, the normal batch norm still need to perform on
    a local batch size, which mean we can't directly do mean/var on a global
    tensor. In order to do a local mean/var, we have to add a new dimention to
    the tensor, so that the ops will not cross the replica boundary. E.g,
    a global tensor with shape [8, x, y] and has 2 local replica, the output of
    this will be [2, 4, x, y], where the first dim is for num of replica, and
    the second dim is for the local batch size. The follow ops can do reduces
    among the local batch dimension.

    Note that this function should only be used under DTensor based strategy,
    and it will use the current strategy in the context to get the number of
    replica.

    Args:
        inputs: Tensor with shape [global_batch_size, ...]

    Returns:
        Tensor with shape [num_replica, local_batch_size, ...]
    """
    input_shape = tf.shape(inputs)
    global_batch_size = input_shape[0]
    num_replica = tf.distribute.get_strategy().num_replicas_in_sync
    local_batch_size = global_batch_size // num_replica
    replica_shape = tf.stack([num_replica, local_batch_size])
    replica_shape = tf.concat([replica_shape, input_shape[1:]], axis=0)
    return tf.reshape(inputs, replica_shape)