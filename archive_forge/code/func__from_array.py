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
def _from_array(z, x, phase=0):
    """Convert array data to BasePauli data."""
    if isinstance(z, np.ndarray) and z.dtype == bool:
        base_z = z
    else:
        base_z = np.asarray(z, dtype=bool)
    if base_z.ndim == 1:
        base_z = base_z.reshape((1, base_z.size))
    elif base_z.ndim != 2:
        raise QiskitError('Invalid Pauli z vector shape.')
    if isinstance(x, np.ndarray) and x.dtype == bool:
        base_x = x
    else:
        base_x = np.asarray(x, dtype=bool)
    if base_x.ndim == 1:
        base_x = base_x.reshape((1, base_x.size))
    elif base_x.ndim != 2:
        raise QiskitError('Invalid Pauli x vector shape.')
    if base_z.shape != base_x.shape:
        raise QiskitError('z and x vectors are different size.')
    dtype = getattr(phase, 'dtype', None)
    base_phase = np.mod(_count_y(base_x, base_z, dtype=dtype) + phase, 4)
    return (base_z, base_x, base_phase)