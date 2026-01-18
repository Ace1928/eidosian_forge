import typing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _ragged_tensor_to_string(string_tensor, summarize):
    """Returns a scalar string tensor with the contents of `string_tensor`.

  Args:
    string_tensor: A potentially ragged tensor with dtype=string.
    summarize: Include only the first and last `summarize` elements of each
      dimension.  If `-1` or `None`, then include all elements.

  Returns:
    A scalar string Tensor.
  """
    if string_tensor.shape.rank == 1:
        pieces = string_tensor
    else:
        pieces = map_fn_lib.map_fn(lambda s: _ragged_tensor_to_string(s, summarize), string_tensor, fn_output_signature=tensor_lib.TensorSpec(None, dtypes.string))
    if summarize not in (-1, None):
        pieces = cond.cond(_nrows(string_tensor) <= 2 * summarize, lambda: pieces, lambda: array_ops.concat([pieces[:summarize], ['...'], pieces[-summarize:]], axis=0))
    return '[' + string_ops.reduce_join(pieces, separator=', ') + ']'