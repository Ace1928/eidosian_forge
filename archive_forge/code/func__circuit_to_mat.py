from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
@staticmethod
def _circuit_to_mat(qc: QuantumCircuit):
    """This creates a nxn matrix corresponding to the given quantum circuit."""
    nq = qc.num_qubits
    mat = np.eye(nq, nq, dtype=bool)
    for instruction in qc.data:
        if instruction.operation.name in ('barrier', 'delay'):
            continue
        if instruction.operation.name == 'cx':
            cb = qc.find_bit(instruction.qubits[0]).index
            tb = qc.find_bit(instruction.qubits[1]).index
            mat[tb, :] = mat[tb, :] ^ mat[cb, :]
            continue
        if instruction.operation.name == 'swap':
            cb = qc.find_bit(instruction.qubits[0]).index
            tb = qc.find_bit(instruction.qubits[1]).index
            mat[[cb, tb]] = mat[[tb, cb]]
            continue
        if getattr(instruction.operation, 'definition', None) is not None:
            other = LinearFunction(instruction.operation.definition)
        else:
            other = LinearFunction(instruction.operation)
        positions = [qc.find_bit(q).index for q in instruction.qubits]
        other = other.extend_with_identity(len(mat), positions)
        mat = np.dot(other.linear.astype(int), mat.astype(int)) % 2
        mat = mat.astype(bool)
    return mat