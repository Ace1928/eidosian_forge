import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _tf_tensor_list_stack(list_, opts):
    """Overload of list_stack that stages a Tensor list write."""
    if opts.element_dtype is None:
        raise ValueError('cannot stack a list without knowing its element type; use set_element_type to annotate it')
    return list_ops.tensor_list_stack(list_, element_dtype=opts.element_dtype)