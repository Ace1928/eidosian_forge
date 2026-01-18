import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING
from cirq.ops import raw_types
class SimpleQubitManager(QubitManager):
    """Allocates a new `CleanQubit`/`BorrowableQubit` for every `qalloc`/`qborrow` request."""

    def __init__(self, prefix: str=''):
        self._clean_id = 0
        self._borrow_id = 0
        self._prefix = prefix

    def qalloc(self, n: int, dim: int=2) -> List['cirq.Qid']:
        self._clean_id += n
        return [CleanQubit(i, dim, self._prefix) for i in range(self._clean_id - n, self._clean_id)]

    def qborrow(self, n: int, dim: int=2) -> List['cirq.Qid']:
        self._borrow_id = self._borrow_id + n
        return [BorrowableQubit(i, dim, self._prefix) for i in range(self._borrow_id - n, self._borrow_id)]

    def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
        for q in qubits:
            good = isinstance(q, CleanQubit) and q.id < self._clean_id
            good |= isinstance(q, BorrowableQubit) and q.id < self._borrow_id
            if not good:
                raise ValueError(f'{q} was not allocated by {self}')