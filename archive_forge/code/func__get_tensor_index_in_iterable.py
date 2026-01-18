from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest
def _get_tensor_index_in_iterable(iterable, t):
    """Returns index of first occurence of `t`, raises ValueError if not found."""
    for i, elem in enumerate(iterable):
        if t is elem:
            return i
    raise ValueError(f'Element `{t!r}` is not found in iterable `{iterable!r}`.')