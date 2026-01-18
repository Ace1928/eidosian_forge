from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _choi_to_kraus(data, input_dim, output_dim, atol=ATOL_DEFAULT):
    """Transform Choi representation to Kraus representation."""
    from scipy import linalg as la
    if is_hermitian_matrix(data, atol=atol):
        w, v = la.schur(data, output='complex')
        w = w.diagonal().real
        if len(w[w < -atol]) == 0:
            kraus = []
            for val, vec in zip(w, v.T):
                if abs(val) > atol:
                    k = np.sqrt(val) * vec.reshape((output_dim, input_dim), order='F')
                    kraus.append(k)
            if not kraus:
                kraus.append(np.zeros((output_dim, input_dim), dtype=complex))
            return (kraus, None)
    mat_u, svals, mat_vh = la.svd(data)
    kraus_l = []
    kraus_r = []
    for val, vec_l, vec_r in zip(svals, mat_u.T, mat_vh.conj()):
        kraus_l.append(np.sqrt(val) * vec_l.reshape((output_dim, input_dim), order='F'))
        kraus_r.append(np.sqrt(val) * vec_r.reshape((output_dim, input_dim), order='F'))
    return (kraus_l, kraus_r)