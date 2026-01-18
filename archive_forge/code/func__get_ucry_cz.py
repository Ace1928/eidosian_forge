from __future__ import annotations
from typing import Callable
import scipy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.two_qubit import (
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit.circuit.library.generalized_gates.ucry import UCRYGate
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate
def _get_ucry_cz(nqubits, angles):
    """
    Get uniformly controlled Ry gate in CZ-Ry as in UCPauliRotGate.
    """
    nangles = len(angles)
    qc = QuantumCircuit(nqubits)
    q_controls = qc.qubits[:-1]
    q_target = qc.qubits[-1]
    if not q_controls:
        if np.abs(angles[0]) > _EPS:
            qc.ry(angles[0], q_target)
    else:
        angles = angles.copy()
        UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
        for i, angle in enumerate(angles):
            if np.abs(angle) > _EPS:
                qc.ry(angle, q_target)
            if not i == len(angles) - 1:
                binary_rep = np.binary_repr(i + 1)
                q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
            else:
                q_contr_index = len(q_controls) - 1
            if i < nangles - 1:
                qc.cz(q_controls[q_contr_index], q_target)
    return qc