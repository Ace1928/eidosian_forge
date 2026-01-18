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
def _sparse_cross_internal(inputs, hashed_output=False, num_buckets=0, hash_key=None, name=None):
    """See gen_sparse_ops.sparse_cross."""
    if not isinstance(inputs, (tuple, list)):
        raise TypeError('Inputs must be a list')
    if not all((isinstance(i, sparse_tensor.SparseTensor) or isinstance(i, tensor_lib.Tensor) for i in inputs)):
        raise TypeError('All inputs must be SparseTensors')
    sparse_inputs = [i for i in inputs if isinstance(i, sparse_tensor.SparseTensor)]
    dense_inputs = [i for i in inputs if not isinstance(i, sparse_tensor.SparseTensor)]
    indices = [sp_input.indices for sp_input in sparse_inputs]
    values = [sp_input.values for sp_input in sparse_inputs]
    shapes = [sp_input.dense_shape for sp_input in sparse_inputs]
    out_type = dtypes.int64 if hashed_output else dtypes.string
    internal_type = dtypes.string
    for i in range(len(values)):
        if values[i].dtype != dtypes.string:
            values[i] = math_ops.cast(values[i], dtypes.int64)
            internal_type = dtypes.int64
    for i in range(len(dense_inputs)):
        if dense_inputs[i].dtype != dtypes.string:
            dense_inputs[i] = math_ops.cast(dense_inputs[i], dtypes.int64)
            internal_type = dtypes.int64
    indices_out, values_out, shape_out = gen_sparse_ops.sparse_cross(indices=indices, values=values, shapes=shapes, dense_inputs=dense_inputs, hashed_output=hashed_output, num_buckets=num_buckets, hash_key=hash_key or _DEFAULT_HASH_KEY, out_type=out_type, internal_type=internal_type, name=name)
    return sparse_tensor.SparseTensor(indices_out, values_out, shape_out)