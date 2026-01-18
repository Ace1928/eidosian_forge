import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
@value.value_equality(approximate=True)
class QuilOneQubitGate(ops.Gate):
    """A QUIL gate representing any single qubit unitary with a DEFGATE and
    2x2 matrix in QUIL.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Inits QuilOneQubitGate.

        Args:
            matrix: The 2x2 unitary matrix for this gate.
        """
        self.matrix = matrix

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq.circuits.quil_output.QuilOneQubitGate(matrix=\n{self.matrix}\n)'

    def _value_equality_values_(self):
        return self.matrix