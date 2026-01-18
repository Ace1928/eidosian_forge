import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
class AcquaintanceOpportunityGate(ops.Gate, ops.InterchangeableQubitsGate):
    """Represents an acquaintance opportunity. An acquaintance opportunity is
    essentially a placeholder in a swap network that may later be replaced with
    a logical gate."""

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits

    def __repr__(self) -> str:
        return f'cirq.contrib.acquaintance.AcquaintanceOpportunityGate(num_qubits={self.num_qubits()!r})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Iterable[str]:
        wire_symbol = '█' if args.use_unicode_characters else 'Acq'
        wire_symbols = (wire_symbol,) * self.num_qubits()
        return wire_symbols

    def num_qubits(self) -> int:
        return self._num_qubits