import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from cirq import protocols
from cirq.ops import raw_types
@functools.total_ordering
class _BaseNamedQid(raw_types.Qid):
    """The base class for `NamedQid` and `NamedQubit`."""
    _name: str
    _dimension: int
    _comp_key: Optional[str] = None
    _hash: Optional[int] = None

    def __getstate__(self):
        state = self.__dict__
        if '_hash' in state or '_comp_key' in state:
            state = state.copy()
            if '_hash' in state:
                del state['_hash']
            if '_comp_key' in state:
                del state['_comp_key']
        return state

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._name, self._dimension))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, _BaseNamedQid):
            return self._name == other._name and self._dimension == other._dimension
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, _BaseNamedQid):
            return self._name != other._name or self._dimension != other._dimension
        return NotImplemented

    def _comparison_key(self):
        if self._comp_key is None:
            self._comp_key = _pad_digits(self._name)
        return self._comp_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'NamedQid':
        return NamedQid(self._name, dimension=dimension)