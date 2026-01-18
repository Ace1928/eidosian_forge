from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction
@staticmethod
def _rowsum_nondeterministic(clifford, accum, row):
    """Updating StabilizerState Clifford in the
        non-deterministic rowsum calculation.
        row and accum are rows in the StabilizerState Clifford."""
    row_phase = clifford.phase[row]
    accum_phase = clifford.phase[accum]
    z = clifford.z
    x = clifford.x
    row_pauli = Pauli((z[row], x[row]))
    accum_pauli = Pauli((z[accum], x[accum]))
    accum_pauli, accum_phase = StabilizerState._rowsum(accum_pauli, accum_phase, row_pauli, row_phase)
    clifford.phase[accum] = accum_phase
    x[accum] = accum_pauli.x
    z[accum] = accum_pauli.z