import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def _get_tensor_name(self, tensor):
    if isinstance(tensor, (tensor_lib.Tensor, variables.Variable)):
        return tensor.name
    elif isinstance(tensor, str):
        return tensor
    else:
        raise TypeError('x_tensor must be a str or tf.Tensor or tf.Variable, but instead has type %s' % type(tensor))