from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def _reverse_clifford(cliff):
    """Reverse qubit order of a Clifford cliff"""
    cliff_cpy = cliff.copy()
    cliff_cpy.stab_z = np.flip(cliff.stab_z, axis=1)
    cliff_cpy.destab_z = np.flip(cliff.destab_z, axis=1)
    cliff_cpy.stab_x = np.flip(cliff.stab_x, axis=1)
    cliff_cpy.destab_x = np.flip(cliff.destab_x, axis=1)
    return cliff_cpy