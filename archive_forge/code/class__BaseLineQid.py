import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
@functools.total_ordering
class _BaseLineQid(ops.Qid):
    """The base class for `LineQid` and `LineQubit`."""
    _x: int
    _dimension: int
    _hash: Optional[int] = None

    def __getstate__(self):
        state = self.__dict__
        if '_hash' in state:
            state = state.copy()
            del state['_hash']
        return state

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._x, self._dimension))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, _BaseLineQid):
            return self._x == other._x and self._dimension == other._dimension
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, _BaseLineQid):
            return self._x != other._x or self._dimension != other._dimension
        return NotImplemented

    def _comparison_key(self):
        return self._x

    @property
    def x(self) -> int:
        return self._x

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'LineQid':
        return LineQid(self._x, dimension)

    def is_adjacent(self, other: 'cirq.Qid') -> bool:
        """Determines if two qubits are adjacent line qubits.

        Args:
            other: `cirq.Qid` to test for adjacency.

        Returns: True iff other and self are adjacent.
        """
        return isinstance(other, _BaseLineQid) and abs(self._x - other._x) == 1

    def neighbors(self, qids: Optional[Iterable[ops.Qid]]=None) -> Set['_BaseLineQid']:
        """Returns qubits that are potential neighbors to this LineQubit

        Args:
            qids: optional Iterable of qubits to constrain neighbors to.
        """
        return {q for q in [self - 1, self + 1] if qids is None or q in qids}

    @abc.abstractmethod
    def _with_x(self, x: int) -> Self:
        """Returns a qubit with the same type but a different value of `x`."""

    def __add__(self, other: Union[int, Self]) -> Self:
        if isinstance(other, _BaseLineQid):
            if self._dimension != other._dimension:
                raise TypeError(f'Can only add LineQids with identical dimension. Got {self._dimension} and {other._dimension}')
            return self._with_x(x=self._x + other._x)
        if not isinstance(other, int):
            raise TypeError(f'Can only add ints and {type(self).__name__}. Instead was {other}')
        return self._with_x(self._x + other)

    def __sub__(self, other: Union[int, Self]) -> Self:
        if isinstance(other, _BaseLineQid):
            if self._dimension != other._dimension:
                raise TypeError(f'Can only subtract LineQids with identical dimension. Got {self._dimension} and {other._dimension}')
            return self._with_x(x=self._x - other._x)
        if not isinstance(other, int):
            raise TypeError(f'Can only subtract ints and {type(self).__name__}. Instead was {other}')
        return self._with_x(self._x - other)

    def __radd__(self, other: int) -> Self:
        return self + other

    def __rsub__(self, other: int) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        return self._with_x(-self._x)

    def __complex__(self) -> complex:
        return complex(self._x)

    def __float__(self) -> float:
        return float(self._x)

    def __int__(self) -> int:
        return int(self._x)