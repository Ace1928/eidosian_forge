from __future__ import annotations
import copy
import sys
from abc import abstractmethod
from numbers import Number, Integral
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _transform_rep
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
def _format_state(self, state, density_matrix=False):
    """Format input state so it is statevector or density matrix"""
    state = np.array(state)
    shape = state.shape
    ndim = state.ndim
    if ndim > 2:
        raise QiskitError('Input state is not a vector or matrix.')
    if ndim == 2:
        if shape[1] != 1 and shape[1] != shape[0]:
            raise QiskitError('Input state is not a vector or matrix.')
        if shape[1] == 1:
            state = np.reshape(state, shape[0])
    if density_matrix and ndim == 1:
        state = np.outer(state, np.transpose(np.conj(state)))
    return state