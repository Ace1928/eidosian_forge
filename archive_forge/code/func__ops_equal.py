from typing import Sequence, Callable
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit.attributes import (
from .optimization_utils import find_next_gate
def _ops_equal(op1, op2):
    """Checks if two operators are equal up to class, data, hyperparameters, and wires"""
    return op1.__class__ is op2.__class__ and op1.data == op2.data and (op1.hyperparameters == op2.hyperparameters) and (op1.wires == op2.wires)