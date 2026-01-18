from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _cross_internal(inputs, hashed_output=False, num_buckets=0, hash_key=None, name=None):
    """Generates feature cross from a list of ragged and dense tensors."""
    if not isinstance(inputs, (tuple, list)):
        raise TypeError('Inputs must be a list')
    if hash_key is None:
        hash_key = _DEFAULT_CROSS_HASH_KEY
    ragged_inputs = []
    sparse_inputs = []
    dense_inputs = []
    input_order = []
    with ops.name_scope(name, 'RaggedCross', inputs):
        for i, t in enumerate(inputs):
            if sparse_tensor.is_sparse(t):
                t = sparse_tensor.SparseTensor.from_value(t)
            else:
                t = ragged_tensor.convert_to_tensor_or_ragged_tensor(t)
            if t.dtype.is_integer:
                t = math_ops.cast(t, dtypes.int64)
            elif t.dtype != dtypes.string:
                raise ValueError('Unexpected dtype for inputs[%d]: %s' % (i, t.dtype))
            if isinstance(t, ragged_tensor.RaggedTensor):
                if t.ragged_rank != 1:
                    raise ValueError('tf.ragged.cross only supports inputs with rank=2')
                ragged_inputs.append(t)
                input_order.append('R')
            elif isinstance(t, sparse_tensor.SparseTensor):
                sparse_inputs.append(t)
                input_order.append('S')
            else:
                dense_inputs.append(t)
                input_order.append('D')
        out_values_type = dtypes.int64 if hashed_output else dtypes.string
        if ragged_inputs and all((t.row_splits.dtype == dtypes.int32 for t in ragged_inputs)):
            out_row_splits_type = dtypes.int32
        else:
            out_row_splits_type = dtypes.int64
        if hash_key > 2 ** 63:
            hash_key -= 2 ** 64
        values_out, splits_out = gen_ragged_array_ops.ragged_cross(ragged_values=[rt.values for rt in ragged_inputs], ragged_row_splits=[rt.row_splits for rt in ragged_inputs], sparse_indices=[st.indices for st in sparse_inputs], sparse_values=[st.values for st in sparse_inputs], sparse_shape=[st.dense_shape for st in sparse_inputs], dense_inputs=dense_inputs, input_order=''.join(input_order), hashed_output=hashed_output, num_buckets=num_buckets, hash_key=hash_key, out_values_type=out_values_type.as_datatype_enum, out_row_splits_type=out_row_splits_type.as_datatype_enum, name=name)
        return ragged_tensor.RaggedTensor.from_row_splits(values_out, splits_out, validate=False)