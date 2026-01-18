from __future__ import annotations
from typing import Iterator, Iterable
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from .annotated_operation import AnnotatedOperation, ControlModifier
from .instruction import Instruction
@staticmethod
def _broadcast_single_argument(qarg: list) -> Iterator[tuple[list, list]]:
    """Expands a single argument.

        For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
    for arg0 in qarg:
        yield ([arg0], [])