from __future__ import annotations
import dataclasses
from typing import Union, List
from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError

        Return the inverse version of itself.

        Implemented as an annotated operation, see  :class:`.AnnotatedOperation`.

        Args:
            annotated: ignored (used for consistency with other inverse methods)

        Returns:
            Inverse version of the given operation.
        