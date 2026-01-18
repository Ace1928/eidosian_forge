import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
@value.value_equality(approximate=True)
class QuilTwoQubitGate(ops.Gate):
    """A two qubit gate represented in QUIL with a DEFGATE and it's 4x4
    unitary matrix.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilTwoQubitGate.

        Args:
            matrix: The 4x4 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self):
        return self.matrix

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilTwoQubitGate(matrix=\n{self.matrix}\n)'