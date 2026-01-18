import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols
@functools.total_ordering
class _BaseGridQid(ops.Qid):
    """The Base class for `GridQid` and `GridQubit`."""
    _row: int
    _col: int
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
            self._hash = hash((self._row, self._col, self._dimension))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, _BaseGridQid):
            return self._row == other._row and self._col == other._col and (self._dimension == other._dimension)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, _BaseGridQid):
            return self._row != other._row or self._col != other._col or self._dimension != other._dimension
        return NotImplemented

    def _comparison_key(self):
        return (self._row, self._col)

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'GridQid':
        return GridQid(self._row, self._col, dimension=dimension)

    def is_adjacent(self, other: 'cirq.Qid') -> bool:
        """Determines if two qubits are adjacent qubits."""
        return isinstance(other, GridQubit) and abs(self._row - other._row) + abs(self._col - other._col) == 1

    def neighbors(self, qids: Optional[Iterable[ops.Qid]]=None) -> Set['_BaseGridQid']:
        """Returns qubits that are potential neighbors to this GridQid

        Args:
            qids: optional Iterable of qubits to constrain neighbors to.
        """
        neighbors = set()
        for q in [self + (0, 1), self + (1, 0), self + (-1, 0), self + (0, -1)]:
            if qids is None or q in qids:
                neighbors.add(q)
        return neighbors

    @abc.abstractmethod
    def _with_row_col(self, row: int, col: int) -> Self:
        """Returns a qid with the same type but a different coordinate."""

    def __complex__(self) -> complex:
        return self._col + 1j * self._row

    def __add__(self, other: Union[Tuple[int, int], Self]) -> Self:
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(f'Can only add GridQids with identical dimension. Got {self.dimension} and {other.dimension}')
            return self._with_row_col(row=self._row + other._row, col=self._col + other._col)
        if not (isinstance(other, (tuple, np.ndarray)) and len(other) == 2 and all((isinstance(x, (int, np.integer)) for x in other))):
            raise TypeError(f'Can only add integer tuples of length 2 to {type(self).__name__}. Instead was {other}')
        return self._with_row_col(row=self._row + other[0], col=self._col + other[1])

    def __sub__(self, other: Union[Tuple[int, int], Self]) -> Self:
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(f'Can only subtract GridQids with identical dimension. Got {self.dimension} and {other.dimension}')
            return self._with_row_col(row=self._row - other._row, col=self._col - other._col)
        if not (isinstance(other, (tuple, np.ndarray)) and len(other) == 2 and all((isinstance(x, (int, np.integer)) for x in other))):
            raise TypeError(f'Can only subtract integer tuples of length 2 to {type(self).__name__}. Instead was {other}')
        return self._with_row_col(row=self._row - other[0], col=self._col - other[1])

    def __radd__(self, other: Tuple[int, int]) -> Self:
        return self + other

    def __rsub__(self, other: Tuple[int, int]) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        return self._with_row_col(row=-self._row, col=-self._col)