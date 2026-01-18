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
def _tp_condition(channel):
    """Return partial tr Choi-matrix eigenvalues for checking if channel is TP"""
    if isinstance(channel, QuantumChannel):
        if not isinstance(channel, Choi):
            channel = Choi(channel)
        choi = channel.data
        dims = tuple(np.sqrt(choi.shape).astype(int))
        shape = dims + dims
        tr_choi = np.trace(np.reshape(choi, shape), axis1=1, axis2=3)
    else:
        unitary = Operator(channel).data
        tr_choi = np.tensordot(unitary, unitary.conj(), axes=(0, 0))
    return np.linalg.eigvalsh(tr_choi - np.eye(len(tr_choi)))