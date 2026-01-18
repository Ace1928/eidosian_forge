import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class VariableSpec(tensor_module.DenseSpec):
    """Describes a tf.Variable.

  A `VariableSpec` provides metadata describing the `tf.Variable` objects
  accepted or returned by TensorFlow 2.x APIs.
  """
    __slots__ = ['trainable', 'alias_id']
    value_type = property(lambda self: ResourceVariable)

    def __init__(self, shape, dtype=dtypes.float32, trainable=True, alias_id=None):
        super(VariableSpec, self).__init__(shape, dtype=dtype)
        self.trainable = trainable
        self.alias_id = alias_id

    def is_compatible_with(self, spec_or_value):
        """Returns True if `spec_or_value` is compatible with this `VariableSpec`.

    `spec_or_value` is considered to be compatible with this `VariableSpec` if

    * `spec_or_value` is a `Variable` or `VariableSpec`,
    * their shapes are compatible,
    * their dtypes are the same,
    * they are both trainable or not trainable.
    * they share the same alias_id if `spec_or_value` is a `VariableSpec`.

    Example:

    >>> v = tf.Variable([1., 2., 3.])
    >>> spec = VariableSpec([None])
    >>> spec.is_compatible_with(v)
    True
    >>> v = tf.Variable(1)
    >>> spec.is_compatible_with(v)
    False

    Args:
      spec_or_value: A VariableSpec or Variable to compare against.

    Returns:
      True if `spec_or_value` is compatible with this `VariableSpec`.
    """
        if not isinstance(spec_or_value, (type(self), self.value_type)):
            return False
        compatible = self.shape.is_compatible_with(spec_or_value.shape) and self.dtype == spec_or_value.dtype and (self.trainable == spec_or_value.trainable)
        if isinstance(spec_or_value, type(self)):
            return compatible and self.alias_id == spec_or_value.alias_id
        return compatible

    @classmethod
    def from_value(cls, value):
        """Creates a `VariableSpec` from the given `Variable`.

    `value`'s shape, dtype, and trainable attributes will be used to create
    the new `VariableSpec`.

    Example:

    >>> v = tf.Variable([1., 2., 3.])
    >>> VariableSpec.from_value(v)
    VariableSpec(shape=(3,), dtype=tf.float32, trainable=True, alias_id=None)

    Args:
      value: A Variable.

    Returns:
      A `VariableSpec` created from `value`.
    """
        return cls(value.shape, dtype=value.dtype, trainable=value.trainable)

    def _to_components(self, value):
        return [value.handle]

    def _from_components(self, components):
        if not isinstance(components, (list, tuple)):
            raise TypeError(f'Components of a ResourceVariable must be a list or tuple, got f{components} instead.')
        if len(components) != 1:
            raise ValueError(f'Components of a ResourceVariable must only contain its resource handle, got f{components} instead.')
        handle = components[0]
        if not isinstance(handle, tensor_module.Tensor) or handle.dtype != dtypes.resource:
            raise ValueError(f'The handle of a ResourceVariable must be a resource tensor, got {handle} instead.')
        return ResourceVariable(trainable=self.trainable, shape=self.shape, dtype=self.dtype, handle=handle)

    @property
    def _component_specs(self):
        return [tensor_module.TensorSpec([], dtypes.DType(dtypes.resource._type_enum, dtypes.HandleData(alias_id=self.alias_id)))]

    def _serialize(self):
        return (self.shape, self.dtype, self.trainable, self.alias_id)

    def is_subtype_of(self, other):
        if type(self) is not type(other):
            return False
        if self.alias_id is None and other.alias_id is None:
            return super().is_subtype_of(other)
        if self.alias_id is None or other.alias_id is None:
            raise NotImplementedError(f"VariableSpec.is_subtype_of doesn't support alias_id=None, got self: {self} and other: {other}.")
        return super().is_subtype_of(other)

    def most_specific_common_supertype(self, others):
        if any((type(self) is not type(other) for other in others)):
            return None
        if self.alias_id is None and all((other.alias_id is None for other in others)):
            return super().most_specific_common_supertype(others)
        if self.alias_id is None or any((other.alias_id is None for other in others)):
            raise NotImplementedError(f"VariableSpec.most_specific_common_supertype doesn't support alias_id=None, got self: {self} and others: {others}.")
        return super().most_specific_common_supertype(others)

    def placeholder_value(self, placeholder_context):
        if placeholder_context.unnest_only:
            return self
        name = self.name or placeholder_context.naming_scope
        context_graph = placeholder_context.context_graph
        if placeholder_context.has_placeholder(self.alias_id):
            variable = placeholder_context.get_placeholder(self.alias_id)
        else:
            spec = tensor_module.TensorSpec([], dtypes.resource)
            spec_context = trace_type.InternalPlaceholderContext(context_graph.outer_graph)
            spec_context.update_naming_scope(name)
            placeholder = spec.placeholder_value(spec_context)
            variable = self._from_components([placeholder])
            if self.alias_id is not None:
                placeholder_context.add_placeholder(self.alias_id, variable)
        placeholder = context_graph.capture(variable.handle, name=name)
        placeholder.op._set_attr('_user_specified_name', attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
        return variable

    def _to_tensors(self, value):
        assert isinstance(value, BaseResourceVariable)
        variable_accessed(value)
        return [value.handle]

    def _cast(self, value, _):
        assert isinstance(value, BaseResourceVariable)
        return value

    def _get_structure(self):
        return PList(PLeaf(), PLeaf(), PLeaf(), PLeaf())

    def __repr__(self):
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype!r}, trainable={self.trainable!r}, alias_id={self.alias_id!r})'

    def __hash__(self):
        return hash((self.shape, self.dtype, self.trainable, self.alias_id))

    def __eq__(self, other):
        return type(self) is type(other) and self.shape == other.shape and (self.dtype == other.dtype) and (self.trainable == other.trainable) and (self.alias_id == other.alias_id)