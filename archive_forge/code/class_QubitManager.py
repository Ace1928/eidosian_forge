import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING
from cirq.ops import raw_types
class QubitManager(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def qalloc(self, n: int, dim: int=2) -> List['cirq.Qid']:
        """Allocate `n` clean qubits, i.e. qubits guaranteed to be in state |0>."""

    @abc.abstractmethod
    def qborrow(self, n: int, dim: int=2) -> List['cirq.Qid']:
        """Allocate `n` dirty qubits, i.e. the returned qubits can be in any state."""

    @abc.abstractmethod
    def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
        """Free pre-allocated clean or dirty qubits managed by this qubit manager."""