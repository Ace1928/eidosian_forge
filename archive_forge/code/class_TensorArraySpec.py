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
@tf_export('TensorArraySpec')
@type_spec_registry.register('tf.TensorArraySpec')
class TensorArraySpec(type_spec.TypeSpec):
    """Type specification for a `tf.TensorArray`."""
    __slots__ = ['_element_shape', '_dtype', '_dynamic_size', '_infer_shape']
    value_type = property(lambda self: TensorArray)

    def __init__(self, element_shape=None, dtype=dtypes.float32, dynamic_size=False, infer_shape=True):
        """Constructs a type specification for a `tf.TensorArray`.

    Args:
      element_shape: The shape of each element in the `TensorArray`.
      dtype: Data type of the `TensorArray`.
      dynamic_size: Whether the `TensorArray` can grow past its initial size.
      infer_shape: Whether shape inference is enabled.
    """
        self._element_shape = tensor_shape.as_shape(element_shape)
        self._dtype = dtypes.as_dtype(dtype)
        self._dynamic_size = dynamic_size
        self._infer_shape = infer_shape

    def is_subtype_of(self, other):
        return isinstance(other, TensorArraySpec) and self._dtype == other._dtype and (self._dynamic_size == other._dynamic_size)

    def most_specific_common_supertype(self, others):
        """Returns the most specific supertype of `self` and `others`.

    Args:
      others: A Sequence of `TypeSpec`.

    Returns `None` if a supertype does not exist.
    """
        if not all((isinstance(other, TensorArraySpec) for other in others)):
            return False
        common_shape = self._element_shape.most_specific_common_supertype((other._element_shape for other in others))
        if common_shape is None:
            return None
        if not all((self._dtype == other._dtype for other in others)):
            return None
        if not all((self._dynamic_size == other._dynamic_size for other in others)):
            return None
        infer_shape = self._infer_shape and all((other._infer_shape for other in others))
        return TensorArraySpec(common_shape, self._dtype, self._dynamic_size, infer_shape)

    def is_compatible_with(self, other):
        if not isinstance(other, type_spec.TypeSpec):
            other = type_spec.type_spec_from_value(other)
        return isinstance(other, TensorArraySpec) and self._dtype.is_compatible_with(other._dtype) and self._element_shape.is_compatible_with(other._element_shape) and (self._dynamic_size == other._dynamic_size)

    def _serialize(self):
        return (self._element_shape, self._dtype, self._dynamic_size, self._infer_shape)

    @property
    def _component_specs(self):
        return [tensor_lib.TensorSpec([], dtypes.variant)]

    def _to_components(self, value):
        if not isinstance(value, TensorArray):
            raise TypeError('Expected value to be a TensorArray, but got: `{}`'.format(type(value)))
        if value.flow is not None and value.flow.dtype == dtypes.variant:
            return [value.flow]
        else:
            with ops.name_scope('convert_tensor_array'):
                flow = list_ops.tensor_list_from_tensor(tensor=value.stack(), element_shape=value.element_shape)
            return [flow]

    def _from_components(self, tensor_list):
        ret = TensorArray(dtype=self._dtype, flow=tensor_list[0], dynamic_size=self._dynamic_size, infer_shape=self._infer_shape)
        ret._implementation._element_shape = [self._element_shape]
        return ret

    @staticmethod
    def from_value(value):
        if not isinstance(value, TensorArray):
            raise TypeError('Expected value to be a TensorArray, but got: `{}`'.format(type(value)))
        return TensorArraySpec(dtype=value.dtype, element_shape=value.element_shape, dynamic_size=value.dynamic_size, infer_shape=value._infer_shape)

    def _to_legacy_output_types(self):
        return self._dtype

    def _to_legacy_output_shapes(self):
        return tensor_shape.TensorShape([self._dynamic_size, self._infer_shape]).concatenate(self._element_shape)

    def _to_legacy_output_classes(self):
        return TensorArray