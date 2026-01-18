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
def _cvxpy_check(name):
    """Check that a supported CVXPY version is installed"""
    _optionals.HAS_CVXPY.require_now(name)
    import cvxpy
    version = cvxpy.__version__
    if version[0] != '1':
        raise MissingOptionalLibraryError('CVXPY >= 1.0', 'diamond_norm', msg=f'Incompatible CVXPY version {version} found.')
    return cvxpy