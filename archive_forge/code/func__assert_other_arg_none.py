from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import tf_utils
def _assert_other_arg_none(arg_name, arg):
    if arg is not None:
        raise ValueError('When `type_spec` is not None, all other args except `name` must be None, but %s is not None.' % arg_name)