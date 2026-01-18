import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumDelay(QuantumInstruction):
    """A built-in ``delay[duration] q0;`` statement."""

    def __init__(self, duration: Expression, qubits: List[Identifier]):
        self.duration = duration
        self.qubits = qubits