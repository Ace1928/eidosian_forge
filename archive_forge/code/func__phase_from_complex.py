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
@staticmethod
def _phase_from_complex(coeff):
    """Return the phase from a label"""
    if np.isclose(coeff, 1):
        return 0
    if np.isclose(coeff, -1j):
        return 1
    if np.isclose(coeff, -1):
        return 2
    if np.isclose(coeff, 1j):
        return 3
    raise QiskitError('Pauli can only be multiplied by 1, -1j, -1, 1j.')