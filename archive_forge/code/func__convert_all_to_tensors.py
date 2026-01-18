import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _convert_all_to_tensors(values, dtype=None, dtype_hint=None):
    """Convert a list of objects to tensors of the same dtype."""
    target_dtype = _get_target_dtype([x for x, _ in values], dtype, dtype_hint)
    convert_behavior = dtype is None
    if convert_behavior:
        return [None if x is None else ops.convert_to_tensor(x, dtype=target_dtype, name=name) for x, name in values]
    else:
        return [None if x is None else math_ops.cast(x, dtype=target_dtype, name=name) for x, name in values]