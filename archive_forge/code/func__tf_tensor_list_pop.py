import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def _tf_tensor_list_pop(list_, i, opts):
    """Overload of list_pop that stages a Tensor list pop."""
    if i is not None:
        raise NotImplementedError('tensor lists only support removing from the end')
    if opts.element_dtype is None:
        raise ValueError('cannot pop from a list without knowing its element type; use set_element_type to annotate it')
    if opts.element_shape is None:
        raise ValueError('cannot pop from a list without knowing its element shape; use set_element_type to annotate it')
    list_out, x = list_ops.tensor_list_pop_back(list_, element_dtype=opts.element_dtype)
    x.set_shape(opts.element_shape)
    return (list_out, x)