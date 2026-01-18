from types import FunctionType
from typing import Type, Union, Callable, Sequence
import pennylane as qml
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.op_math import Adjoint
def _check_position(position):
    """Checks the position argument to determine if an operation or list of operations was provided."""
    not_op = False
    req_ops = False
    if isinstance(position, list):
        req_ops = position.copy()
        for operation in req_ops:
            try:
                if Operation not in operation.__bases__:
                    not_op = True
            except AttributeError:
                not_op = True
    elif not isinstance(position, list):
        try:
            if Operation in position.__bases__:
                req_ops = [position]
            else:
                not_op = True
        except AttributeError:
            not_op = True
    return (not_op, req_ops)