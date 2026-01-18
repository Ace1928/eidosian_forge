from __future__ import annotations
import logging
import numpy as np
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel import Choi, SuperOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.utils import optionals as _optionals
def _cp_condition(channel):
    """Return Choi-matrix eigenvalues for checking if channel is CP"""
    if isinstance(channel, QuantumChannel):
        if not isinstance(channel, Choi):
            channel = Choi(channel)
        return np.linalg.eigvalsh(channel.data)
    unitary = Operator(channel).data
    return np.tensordot(unitary, unitary.conj(), axes=([0, 1], [0, 1])).real