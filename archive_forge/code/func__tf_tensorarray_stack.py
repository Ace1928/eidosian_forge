import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _tf_tensorarray_stack(list_):
    """Overload of list_stack that stages a TensorArray stack."""
    return list_.stack()