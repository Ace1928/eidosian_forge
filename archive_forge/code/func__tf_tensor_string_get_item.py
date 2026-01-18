import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _tf_tensor_string_get_item(target, i):
    """Overload of get_item that stages a Tensor string read."""
    x = gen_string_ops.substr(target, i, 1)
    return x