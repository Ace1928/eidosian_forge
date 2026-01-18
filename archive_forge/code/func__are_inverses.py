from typing import Sequence, Callable
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit.attributes import (
from .optimization_utils import find_next_gate
def _are_inverses(op1, op2):
    """Checks if two operators are inverses of each other

    Args:
        op1 (~.Operator)
        op2 (~.Operator)

    Returns:
        Bool
    """
    if op1 in self_inverses and op1.name == op2.name:
        return True
    if isinstance(op1, Adjoint) and _ops_equal(op1.base, op2):
        return True
    if isinstance(op2, Adjoint) and _ops_equal(op2.base, op1):
        return True
    return False