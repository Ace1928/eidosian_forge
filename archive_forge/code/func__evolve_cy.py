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
def _evolve_cy(base_pauli, qctrl, qtrgt):
    """Update P -> CY.P.CY"""
    x1 = base_pauli._x[:, qctrl].copy()
    x2 = base_pauli._x[:, qtrgt].copy()
    z2 = base_pauli._z[:, qtrgt].copy()
    base_pauli._x[:, qtrgt] ^= x1
    base_pauli._z[:, qtrgt] ^= x1
    base_pauli._z[:, qctrl] ^= np.logical_xor(x2, z2)
    base_pauli._phase += x1 + 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli