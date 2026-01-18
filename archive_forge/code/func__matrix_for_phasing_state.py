import dataclasses
import cirq
import numpy as np
from cirq import ops, qis, protocols
def _matrix_for_phasing_state(num_qubits, phase_state, phase):
    matrix = qis.eye_tensor((2,) * num_qubits, dtype=np.complex128)
    matrix = matrix.reshape((2 ** num_qubits, 2 ** num_qubits))
    matrix[phase_state, phase_state] = phase
    print(num_qubits, phase_state, phase)
    print(matrix)
    return matrix