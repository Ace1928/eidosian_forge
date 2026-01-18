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
def _input_formatter(obj, fallback_class, func_name, arg_name):
    """Formatting function for input conversion"""
    if obj is None:
        return obj
    if isinstance(obj, QuantumChannel):
        return obj
    if hasattr(obj, 'to_quantumchannel'):
        return obj.to_quantumchannel()
    if hasattr(obj, 'to_channel'):
        return obj.to_channel()
    if isinstance(obj, (Gate, BaseOperator)):
        return Operator(obj)
    if hasattr(obj, 'to_operator'):
        return obj.to_operator()
    raise TypeError(f'invalid type supplied to {arg_name} of {func_name}. A {fallback_class.__name__} is best.')