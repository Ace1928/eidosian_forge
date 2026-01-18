import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class Dict(trace.TraceType, serialization.Serializable):
    """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """

    def __init__(self, mapping: PythonDict[Hashable, trace.TraceType], placeholder_type: Optional[Type[Any]]=None):
        self.mapping = mapping
        self._placeholder_type = placeholder_type

    def _has_same_structure(self, other):
        if not isinstance(other, Dict):
            return False
        return self.mapping.keys() == other.mapping.keys()

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        """See base class."""
        if not self._has_same_structure(other):
            return False
        return all((self.mapping[key].is_subtype_of(other.mapping[key]) for key in self.mapping))

    def most_specific_common_supertype(self, types: Sequence[trace.TraceType]) -> Optional['Dict']:
        """See base class."""
        if not all((self._has_same_structure(other) for other in types)):
            return None
        new_mapping = {}
        for key in self.mapping.keys():
            common = self.mapping[key].most_specific_common_supertype([other.mapping[key] for other in types])
            if common is None:
                return None
            else:
                new_mapping[key] = common
        return Dict(new_mapping, self._placeholder_type)

    @classmethod
    def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedDict]:
        return default_types_pb2.SerializedDict

    @classmethod
    def experimental_from_proto(cls, proto: default_types_pb2.SerializedDict) -> 'Dict':
        return Dict({Literal.experimental_from_proto(k).value: serialization.deserialize(v) for k, v in zip(proto.keys, proto.values)})

    def experimental_as_proto(self) -> default_types_pb2.SerializedDict:
        return default_types_pb2.SerializedDict(keys=[Literal(k).experimental_as_proto() for k in self.mapping.keys()], values=[serialization.serialize(v) for v in self.mapping.values()])

    def placeholder_value(self, placeholder_context) -> Any:
        if self._placeholder_type is None:
            raise ValueError('Can not generate placeholder value for Dict with unspecified placeholder_type. Note: placeholder_type is lost during serialization.')
        attribute_placeholders = [(key, value.placeholder_value(placeholder_context)) for key, value in self.mapping.items()]
        if self._placeholder_type is collections.defaultdict:
            return dict(attribute_placeholders)
        return self._placeholder_type(attribute_placeholders)

    def _to_tensors(self, value: Any):
        assert isinstance(value, collections.abc.Mapping)
        flattened_values = []
        for key in sorted(self.mapping.keys()):
            comp_value, comp_type = (value[key], self.mapping[key])
            flattened_values.extend(comp_type._to_tensors(comp_value))
        return flattened_values

    def _from_tensors(self, tensors):
        if self._placeholder_type is None:
            raise ValueError('Packing serialized Dict is not supported.')
        sorted_traversal = {key: self.mapping[key]._from_tensors(tensors) for key in sorted(self.mapping)}
        if self._placeholder_type is collections.defaultdict:
            return {key: sorted_traversal[key] for key in self.mapping}
        return self._placeholder_type(((key, sorted_traversal[key]) for key in self.mapping))

    def _flatten(self) -> PythonList[trace.TraceType]:
        flattened_types = []
        for key in sorted(self.mapping.keys()):
            flattened_types.extend(self.mapping[key]._flatten())
        return flattened_types

    def _cast(self, value: Any, casting_context) -> Any:
        assert isinstance(value, collections.abc.Mapping), f'Can not cast {value!r} to a Dict type.'
        assert set(value.keys()) == set(self.mapping.keys()), f'{value!r} has different keys with the TraceType {self!r}.'
        casted_values, was_casted = cast_and_return_whether_casted(self.mapping.values(), [value[k] for k in self.mapping.keys()], casting_context)
        if was_casted:
            return self._placeholder_type(**{k: v for k, v in zip(self.mapping.keys(), casted_values)})
        else:
            return value

    def __eq__(self, other) -> bool:
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, Dict):
            return False
        return self.mapping == other.mapping

    def __hash__(self) -> int:
        return hash(frozenset(self.mapping.keys()))

    def __repr__(self):
        return f'{self.__class__.__name__}(mapping={self.mapping!r})'