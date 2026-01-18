import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('TypeSpec', v1=['TypeSpec', 'data.experimental.Structure'])
class TypeSpec(internal.TypeSpec, trace.TraceType, trace_type.Serializable, metaclass=abc.ABCMeta):
    """Specifies a TensorFlow value type.

  A `tf.TypeSpec` provides metadata describing an object accepted or returned
  by TensorFlow APIs.  Concrete subclasses, such as `tf.TensorSpec` and
  `tf.RaggedTensorSpec`, are used to describe different value types.

  For example, `tf.function`'s `input_signature` argument accepts a list
  (or nested structure) of `TypeSpec`s.

  Creating new subclasses of `TypeSpec` (outside of TensorFlow core) is not
  currently supported.  In particular, we may make breaking changes to the
  private methods and properties defined by this base class.

  Example:

  >>> spec = tf.TensorSpec(shape=[None, None], dtype=tf.int32)
  >>> @tf.function(input_signature=[spec])
  ... def double(x):
  ...   return x * 2
  >>> double(tf.constant([[1, 2], [3, 4]]))
  <tf.Tensor: shape=(2, 2), dtype=int32,
      numpy=array([[2, 4], [6, 8]], dtype=int32)>
  """
    __slots__ = CACHED_FIXED_PROPERTIES

    @abc.abstractproperty
    def value_type(self):
        """The Python type for values that are compatible with this TypeSpec.

    In particular, all values that are compatible with this TypeSpec must be an
    instance of this type.
    """
        raise NotImplementedError('%s.value_type' % type(self).__name__)

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        """Returns True if `self` is a subtype of `other`.

    Implements the tf.types.experimental.func.TraceType interface.

    If not overridden by a subclass, the default behavior is to assume the
    TypeSpec is covariant upon attributes that implement TraceType and
    invariant upon rest of the attributes as well as the structure and type
    of the TypeSpec.

    Args:
      other: A TraceType object.
    """
        if type(self) is not type(other):
            return False
        is_subtype = True

        def check_attribute(attribute_self, attribute_other):
            nonlocal is_subtype
            if not is_subtype:
                return
            if isinstance(attribute_self, trace.TraceType):
                if not attribute_self.is_subtype_of(attribute_other):
                    is_subtype = False
                    return
            elif attribute_self != attribute_other:
                is_subtype = False
        try:
            nest.map_structure(check_attribute, self._serialize(), other._serialize())
        except (ValueError, TypeError):
            return False
        return is_subtype

    def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional['TypeSpec']:
        """Returns the most specific supertype TypeSpec  of `self` and `others`.

    Implements the tf.types.experimental.func.TraceType interface.

    If not overridden by a subclass, the default behavior is to assume the
    TypeSpec is covariant upon attributes that implement TraceType and
    invariant upon rest of the attributes as well as the structure and type
    of the TypeSpec.

    Args:
      others: A sequence of TraceTypes.
    """
        if any((type(self) is not type(other) for other in others)):
            return None
        has_supertype = True

        def make_supertype_attribute(attribute_self, *attribute_others):
            nonlocal has_supertype
            if not has_supertype:
                return
            if isinstance(attribute_self, trace.TraceType):
                attribute_supertype = attribute_self.most_specific_common_supertype(attribute_others)
                if attribute_supertype is None:
                    has_supertype = False
                    return
                return attribute_supertype
            else:
                if not all((attribute_self == attribute_other for attribute_other in attribute_others)):
                    has_supertype = False
                    return
                return attribute_self
        try:
            serialized_supertype = nest.map_structure(make_supertype_attribute, self._serialize(), *(o._serialize() for o in others))
        except (ValueError, TypeError):
            return None
        return self._deserialize(serialized_supertype) if has_supertype else None

    @classmethod
    def experimental_type_proto(cls) -> Type[struct_pb2.TypeSpecProto]:
        """Returns the type of proto associated with TypeSpec serialization.

    Do NOT override for custom non-TF types.
    """
        return struct_pb2.TypeSpecProto

    @classmethod
    def experimental_from_proto(cls, proto: struct_pb2.TypeSpecProto) -> 'TypeSpec':
        """Returns a TypeSpec instance based on the serialized proto.

    Do NOT override for custom non-TF types.

    Args:
      proto: Proto generated using 'experimental_as_proto'.
    """
        return nested_structure_coder.decode_proto(struct_pb2.StructuredValue(type_spec_value=proto))

    def experimental_as_proto(self) -> struct_pb2.TypeSpecProto:
        """Returns a proto representation of the TypeSpec instance.

    Do NOT override for custom non-TF types.
    """
        return nested_structure_coder.encode_structure(self).type_spec_value

    @doc_controls.do_not_doc_inheritable
    def placeholder_value(self, placeholder_context):
        """Value used for tracing a function signature with this TraceType.

    WARNING: Do not override.

    Args:
      placeholder_context: A class container for context information when
        creating a placeholder value.

    Returns:
      A `CompositeTensor` placeholder whose components are recursively composed
        of placeholders themselves.
    """
        if placeholder_context.unnest_only:
            return self
        component_placeholders = nest.map_structure(lambda x: x.placeholder_value(placeholder_context), self._component_specs)
        return self._from_components(component_placeholders)

    def _to_tensors(self, value):
        tensors = []
        nest.map_structure(lambda spec, v: tensors.extend(spec._to_tensors(v)), self._component_specs, self._to_components(value))
        return tensors

    def _from_tensors(self, tensors):
        components = nest.map_structure(lambda spec: spec._from_tensors(tensors), self._component_specs)
        return self._from_components(components)

    def _flatten(self):
        flat = []
        nest.map_structure(lambda spec: flat.extend(spec._flatten()), self._component_specs)
        return flat

    def _cast(self, value, casting_context):
        if casting_context.allow_specs and isinstance(value, TypeSpec):
            assert value.is_subtype_of(self), f'Can not cast {value!r} to {self!r}'
            return self
        did_cast = False

        def cast_fn(spec, v):
            casted_v = spec._cast(v, casting_context)
            if casted_v is not v:
                nonlocal did_cast
                did_cast = True
            return casted_v
        cast_components = nest.map_structure(cast_fn, self._component_specs, self._to_components(value))
        if did_cast:
            return self._from_components(cast_components)
        else:
            return value

    def is_compatible_with(self, spec_or_value):
        """Returns true if `spec_or_value` is compatible with this TypeSpec.

    Prefer using "is_subtype_of" and "most_specific_common_supertype" wherever
    possible.

    Args:
      spec_or_value: A TypeSpec or TypeSpec associated value to compare against.
    """
        if not isinstance(spec_or_value, TypeSpec):
            spec_or_value = type_spec_from_value(spec_or_value)
        if type(self) is not type(spec_or_value):
            return False
        return self.__is_compatible(self._serialize(), spec_or_value._serialize())

    @deprecation.deprecated(None, 'Use most_specific_common_supertype instead.')
    def most_specific_compatible_type(self, other: 'TypeSpec') -> 'TypeSpec':
        """Returns the most specific TypeSpec compatible with `self` and `other`.

    Deprecated. Please use `most_specific_common_supertype` instead.
    Do not override this function.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
        result = self.most_specific_common_supertype([other])
        if result is None:
            raise ValueError('No TypeSpec is compatible with both %s and %s' % (self, other))
        return result

    def _with_tensor_ranks_only(self) -> 'TypeSpec':
        """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only tensor rank (and not
      the dimension sizes for individual axes).
    """

        def relax(value):
            if isinstance(value, TypeSpec):
                return value._with_tensor_ranks_only()
            elif isinstance(value, tensor_shape.TensorShape) and value.rank is not None:
                return tensor_shape.TensorShape([None] * value.rank)
            else:
                return value
        return self._deserialize(nest.map_structure(relax, self._serialize()))

    def _without_tensor_names(self) -> 'TypeSpec':
        """Returns a TypeSpec compatible with `self`, with tensor names removed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where the name of any
      `TensorSpec` is set to `None`.
    """

        def rename(value):
            if isinstance(value, TypeSpec):
                return value._without_tensor_names()
            return value
        return self._deserialize(nest.map_structure(rename, self._serialize()))

    @abc.abstractmethod
    def _to_components(self, value):
        """Encodes `value` as a nested structure of `Tensor` or `CompositeTensor`.

    Args:
      value: A value compatible with this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A nested structure of `tf.Tensor` or `tf.CompositeTensor` compatible with
      `self._component_specs`, which can be used to reconstruct `value`.
    """
        raise NotImplementedError('%s._to_components()' % type(self).__name__)

    @abc.abstractmethod
    def _from_components(self, components):
        """Reconstructs a value from a nested structure of Tensor/CompositeTensor.

    Args:
      components: A nested structure of `tf.Tensor` or `tf.CompositeTensor`,
        compatible with `self._component_specs`.  (Caller is responsible for
        ensuring compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
        raise NotImplementedError('%s._from_components()' % type(self).__name__)

    @abc.abstractproperty
    def _component_specs(self):
        """A nested structure of TypeSpecs for this type's components.

    Returns:
      A nested structure describing the component encodings that are returned
      by this TypeSpec's `_to_components` method.  In particular, for a
      TypeSpec `spec` and a compatible value `value`:

      ```
      nest.map_structure(lambda t, c: assert t.is_compatible_with(c),
                         spec._component_specs, spec._to_components(value))
      ```
    """
        raise NotImplementedError('%s._component_specs()' % type(self).__name__)

    def _to_tensor_list(self, value) -> List['core_types.Symbol']:
        """Encodes `value` as a flat list of `tf.Tensor`.

    By default, this just flattens `self._to_components(value)` using
    `nest.flatten`.  However, subclasses may override this to return a
    different tensor encoding for values.  In particular, some subclasses
    of `BatchableTypeSpec` override this method to return a "boxed" encoding
    for values, which then can be batched or unbatched.  See
    `BatchableTypeSpec` for more details.

    Args:
      value: A value with compatible this `TypeSpec`.  (Caller is responsible
        for ensuring compatibility.)

    Returns:
      A list of `tf.Tensor`, compatible with `self._flat_tensor_specs`, which
      can be used to reconstruct `value`.
    """
        return nest.flatten(self._to_components(value), expand_composites=True)

    def _from_tensor_list(self, tensor_list: List['core_types.Symbol']) -> Any:
        """Reconstructs a value from a flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.

    Returns:
      A value that is compatible with this `TypeSpec`.

    Raises:
      ValueError: If `tensor_list` is not compatible with
      `self._flat_tensor_specs`.
    """
        self.__check_tensor_list(tensor_list)
        return self._from_compatible_tensor_list(tensor_list)

    def _from_compatible_tensor_list(self, tensor_list: List['core_types.Symbol']) -> Any:
        """Reconstructs a value from a compatible flat list of `tf.Tensor`.

    Args:
      tensor_list: A flat list of `tf.Tensor`, compatible with
        `self._flat_tensor_specs`.  (Caller is responsible for ensuring
        compatibility.)

    Returns:
      A value that is compatible with this `TypeSpec`.
    """
        return self._from_components(nest.pack_sequence_as(self._component_specs, tensor_list, expand_composites=True))

    @property
    def _flat_tensor_specs(self):
        """A list of TensorSpecs compatible with self._to_tensor_list(v)."""
        return nest.flatten(self._component_specs, expand_composites=True)

    @abc.abstractmethod
    def _serialize(self):
        """Returns a nested tuple containing the state of this TypeSpec.

    The serialization may contain the following value types: boolean,
    integer, string, float, None, `TensorSpec`, `tf.TensorShape`, `tf.DType`,
    `np.ndarray`, `TypeSpec`, and nested tuples, namedtuples, dicts, and
    OrderedDicts of any of the above.

    This method is used to provide default definitions for: equality
    testing (__eq__, __ne__), hashing (__hash__), pickling (__reduce__),
    string representation (__repr__), `self.is_compatible_with()`,
    `self.most_specific_compatible_type()`, and protobuf serialization
    (e.g. TensorInfo and StructuredValue).
    """
        raise NotImplementedError('%s._serialize()' % type(self).__name__)

    @classmethod
    def _deserialize(cls, serialization):
        """Reconstructs a TypeSpec from a value returned by `serialize`.

    Args:
      serialization: A value returned by _serialize.  In some contexts,
        `namedtuple`s in `serialization` may not have the identical type that
        was returned by `_serialize` (but its type will still be a `namedtuple`
        type with the same type name and field names).  For example, the code
        that loads a SavedModel does not have access to the original
        `namedtuple` type, so it dynamically creates a new `namedtuple` type
        with the same type name and field names as the original one.  If
        necessary, you can check `serialization` for these duck-typed
        `nametuple` types, and restore them to the original type. (E.g., this
        would be necessary if you rely on type checks such as `isinstance` for
        this `TypeSpec`'s member variables).

    Returns:
      A `TypeSpec` of type `cls`.
    """
        return cls(*serialization)

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and self.__get_cmp_key() == other.__get_cmp_key()

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.__get_cmp_key())

    def __reduce__(self):
        return (type(self), self._serialize())

    def __repr__(self) -> str:
        return '%s%r' % (type(self).__name__, self._serialize())

    def _to_legacy_output_types(self):
        raise NotImplementedError('%s._to_legacy_output_types()' % type(self).__name__)

    def _to_legacy_output_shapes(self):
        raise NotImplementedError('%s._to_legacy_output_shapes()' % type(self).__name__)

    def _to_legacy_output_classes(self):
        return self.value_type

    def __tf_tracing_type__(self, context: trace.TracingContext) -> trace.TraceType:
        return self

    def __check_tensor_list(self, tensor_list):
        """Raises an exception if tensor_list incompatible w/ flat_tensor_specs."""
        expected = self._flat_tensor_specs
        specs = [type_spec_from_value(t) for t in tensor_list]
        if len(specs) != len(expected):
            raise ValueError(f'Cannot create a {self.value_type.__name__} from the tensor list because the TypeSpec expects {len(expected)} items, but the provided tensor list has {len(specs)} items.')
        for i, (s1, s2) in enumerate(zip(specs, expected)):
            if not s1.is_compatible_with(s2):
                raise ValueError(f'Cannot create a {self.value_type.__name__} from the tensor list because item {i} ({tensor_list[i]!r}) is incompatible with the expected TypeSpec {s2}.')

    def __get_cmp_key(self):
        """Returns a hashable eq-comparable key for `self`."""
        if not hasattr(self, _CACHED_CMP_KEY):
            setattr(self, _CACHED_CMP_KEY, (type(self), self.__make_cmp_key(self._serialize())))
        return getattr(self, _CACHED_CMP_KEY)

    def __make_cmp_key(self, value):
        """Converts `value` to a hashable key."""
        if isinstance(value, (int, float, bool, np.generic, dtypes.DType, TypeSpec, tensor_shape.TensorShape)):
            return value
        if isinstance(value, compat.bytes_or_text_types):
            return value
        if value is None:
            return value
        if isinstance(value, dict):
            return tuple([tuple([self.__make_cmp_key(key), self.__make_cmp_key(value[key])]) for key in sorted(value.keys())])
        if isinstance(value, tuple):
            return tuple([self.__make_cmp_key(v) for v in value])
        if isinstance(value, list):
            return (list, tuple([self.__make_cmp_key(v) for v in value]))
        if isinstance(value, np.ndarray):
            return (np.ndarray, value.shape, TypeSpec.__nested_list_to_tuple(value.tolist()))
        raise ValueError(f'Cannot generate a hashable key for {self} because the _serialize() method returned an unsupproted value of type {type(value)}')

    @staticmethod
    def __nested_list_to_tuple(value):
        """Converts a nested list to a corresponding nested tuple."""
        if isinstance(value, list):
            return tuple((TypeSpec.__nested_list_to_tuple(v) for v in value))
        return value

    @staticmethod
    def __same_types(a, b):
        """Returns whether a and b have the same type, up to namedtuple equivalence.

    Consistent with tf.nest.assert_same_structure(), two namedtuple types
    are considered the same iff they agree in their class name (without
    qualification by module name) and in their sequence of field names.
    This makes namedtuples recreated by nested_structure_coder compatible with
    their original Python definition.

    Args:
      a: a Python object.
      b: a Python object.

    Returns:
      A boolean that is true iff type(a) and type(b) are the same object
      or equivalent namedtuple types.
    """
        if nest.is_namedtuple(a) and nest.is_namedtuple(b):
            return nest.same_namedtuples(a, b)
        else:
            return type(a) is type(b)

    @staticmethod
    def __is_compatible(a, b):
        """Returns true if the given type serializations compatible."""
        if isinstance(a, TypeSpec):
            return a.is_compatible_with(b)
        if not TypeSpec.__same_types(a, b):
            return False
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all((TypeSpec.__is_compatible(x, y) for x, y in zip(a, b)))
        if isinstance(a, dict):
            return len(a) == len(b) and sorted(a.keys()) == sorted(b.keys()) and all((TypeSpec.__is_compatible(a[k], b[k]) for k in a.keys()))
        if isinstance(a, (tensor_shape.TensorShape, dtypes.DType)):
            return a.is_compatible_with(b)
        return a == b