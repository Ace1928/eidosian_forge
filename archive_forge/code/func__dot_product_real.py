from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
def _dot_product_real(bra, ket, num_wires):
    """Helper for calculating the inner product for adjoint differentiation."""
    sum_axes = tuple(range(1, num_wires + 1))
    return qml.math.real(qml.math.sum(qml.math.conj(bra) * ket, axis=sum_axes))