from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _transform_from_pauli(data, num_qubits):
    """Change of basis of bipartite matrix representation."""
    basis_mat = np.array([[1, 0, 0, 1], [0, 1, 1j, 0], [0, 1, -1j, 0], [1, 0j, 0, -1]], dtype=complex)
    cob = basis_mat
    for _ in range(num_qubits - 1):
        dim = int(np.sqrt(len(cob)))
        cob = np.reshape(np.transpose(np.reshape(np.kron(basis_mat, cob), (2, 2, dim, dim, 4, dim * dim)), (0, 2, 1, 3, 4, 5)), (4 * dim * dim, 4 * dim * dim))
    return np.dot(np.dot(cob, data), cob.conj().T) / 2 ** num_qubits