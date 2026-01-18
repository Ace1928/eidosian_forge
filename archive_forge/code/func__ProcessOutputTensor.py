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
def _ProcessOutputTensor(self, val):
    """Process an output tensor of a conditional branch."""
    real_val = val
    if val.name not in self._values:
        self._values.add(val.name)
        if self._outer_context:
            real_val = self._outer_context.AddValue(val)
            self._values.add(real_val.name)
            self._external_values[real_val.name] = real_val
        real_val = _SwitchRefOrTensor(real_val, self._pred)[self._branch]
        self._external_values[val.name] = real_val
    else:
        external_val = self._external_values.get(val.name)
        if external_val is not None:
            real_val = external_val
    return real_val