import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
def _convert_to_sparse_tensors(sp_inputs):
    """Convert `sp_inputs` to `SparseTensor` objects and return them.

  Args:
    sp_inputs: `list` or `tuple` of `SparseTensor` or `SparseTensorValue`
      objects.

  Returns:
    `sp_inputs` converted to `SparseTensor` objects.

  Raises:
    ValueError: if any item in `sp_inputs` is neither `SparseTensor` nor
      `SparseTensorValue`.
  """
    if isinstance(sp_inputs, list):
        return [_convert_to_sparse_tensor(sp_input) for sp_input in sp_inputs]
    if isinstance(sp_inputs, tuple):
        return (_convert_to_sparse_tensor(sp_input) for sp_input in sp_inputs)
    raise TypeError('Inputs must be a list or tuple.')