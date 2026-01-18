import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _tf_tensor_list_get_item(target, i, opts):
    """Overload of get_item that stages a Tensor list read."""
    if opts.element_dtype is None:
        raise ValueError('cannot retrieve from a list without knowing its element type; use set_element_type to annotate it')
    x = list_ops.tensor_list_get_item(target, i, element_dtype=opts.element_dtype)
    return x