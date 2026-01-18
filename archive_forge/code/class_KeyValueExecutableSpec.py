import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
@dataclass(frozen=True)
class KeyValueExecutableSpec(ExecutableSpec):
    """A generic executable spec whose metadata is a list of key-value pairs.

    The key-value pairs define an implicit data schema. Consider defining a problem-specific
    subclass of `ExecutableSpec` instead of using this class to realize the benefits of having
    an explicit schema.

    See Also:
        `KeyValueExecutableSpec.from_dict` will use a dictionary to populate `key_value_pairs`.

    Args:
        executable_family: A unique name to group executables.
        key_value_pairs: A tuple of key-value pairs. The keys should be strings but the values
            can be any immutable object. Note that the order of the key-value pairs does NOT matter
            when comparing two objects.
    """
    executable_family: str
    key_value_pairs: Tuple[Tuple[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.key_value_pairs)

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, executable_family: str) -> 'KeyValueExecutableSpec':
        return cls(executable_family=executable_family, key_value_pairs=tuple(((k, v) for k, v in d.items())))

    @classmethod
    def _from_json_dict_(cls, executable_family: str, key_value_pairs: List[List[Union[str, Any]]], **kwargs) -> 'KeyValueExecutableSpec':
        return cls(executable_family=executable_family, key_value_pairs=tuple(((k, v) for k, v in key_value_pairs)))

    def __repr__(self) -> str:
        return cirq._compat.dataclass_repr(self, namespace='cirq_google')

    def __eq__(self, other):
        return self.executable_family == other.executable_family and dict(self.key_value_pairs) == dict(other.key_value_pairs)