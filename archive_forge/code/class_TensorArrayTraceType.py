import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
class TensorArrayTraceType(trace.TraceType):
    """Represents TraceType of TensorArray."""

    def __init__(self, value):
        self._value = value

    def is_subtype_of(self, other):
        return self == other

    def most_specific_common_supertype(self, types):
        return self if all((self == other for other in types)) else None

    def placeholder_value(self, placeholder_context):
        return self._value

    def _flatten(self):
        return [tensor_lib.TensorSpec([], dtypes.variant)]

    def _from_tensors(self, tensors):
        return next(tensors)

    def __eq__(self, other):
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, TensorArrayTraceType):
            return False
        return self._value is other._value

    def __hash__(self):
        return id(self._value)

    def __repr__(self):
        return f'{self.__class__.__name__}(value={self._value!r})'