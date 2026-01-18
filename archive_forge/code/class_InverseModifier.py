from __future__ import annotations
import dataclasses
from typing import Union, List
from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError
@dataclasses.dataclass
class InverseModifier(Modifier):
    """Inverse modifier: specifies that the operation is inverted."""
    pass