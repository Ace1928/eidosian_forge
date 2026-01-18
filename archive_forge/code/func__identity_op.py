from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
@lru_cache(maxsize=None)
def _identity_op(num_qubits):
    """Cached identity matrix"""
    return Operator(np.eye(2 ** num_qubits), input_dims=(2,) * num_qubits, output_dims=(2,) * num_qubits)