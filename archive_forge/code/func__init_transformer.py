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
@classmethod
def _init_transformer(cls, data):
    """Convert input into a QuantumChannel subclass object or Operator object"""
    if isinstance(data, QuantumChannel):
        return data
    if hasattr(data, 'to_quantumchannel'):
        return data.to_quantumchannel()
    if hasattr(data, 'to_channel'):
        return data.to_channel()
    return Operator(data)