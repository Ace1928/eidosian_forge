import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _NextIteration(tensor, name=None):
    tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
    if isinstance(tensor, tensor_lib.Tensor):
        if tensor.dtype._is_ref_dtype:
            return ref_next_iteration(tensor, name=name)
        else:
            return next_iteration(tensor, name=name)
    elif isinstance(tensor, composite_tensor.CompositeTensor):
        return nest.map_structure(_NextIteration, tensor, expand_composites=True)
    else:
        raise TypeError(f"'tensor' must be a Tensor or CompositeTensor. Received: {type(tensor)}.")