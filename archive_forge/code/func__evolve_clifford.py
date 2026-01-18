from __future__ import annotations
import copy
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
def _evolve_clifford(self, other, qargs=None, frame='h'):
    """Heisenberg picture evolution of a Pauli by a Clifford."""
    if frame == 's':
        adj = other
    else:
        adj = other.adjoint()
    if qargs is None:
        qargs_ = slice(None)
    else:
        qargs_ = list(qargs)
    from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
    num_paulis = self._x.shape[0]
    ret = self.copy()
    ret._x[:, qargs_] = False
    ret._z[:, qargs_] = False
    idx = np.concatenate((self._x[:, qargs_], self._z[:, qargs_]), axis=1)
    for idx_, row in zip(idx.T, PauliList.from_symplectic(z=adj.z, x=adj.x, phase=2 * adj.phase)):
        if idx_.any():
            if np.sum(idx_) == num_paulis:
                ret.compose(row, qargs=qargs, inplace=True)
            else:
                ret[idx_] = ret[idx_].compose(row, qargs=qargs)
    return ret