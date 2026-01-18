from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
def _adjoint_jacobian_state(tape: QuantumTape):
    """Calculate the full jacobian for a circuit that returns the state.

    Args:
        tape (QuantumTape): the circuit we wish to differentiate

    Returns:
        TensorLike: the full jacobian.

    See ``adjoint_jacobian.md`` for details on the algorithm.
    """
    jacobian = []
    has_state_prep = isinstance(tape[0], qml.operation.StatePrepBase)
    state = create_initial_state(tape.wires, tape[0] if has_state_prep else None)
    param_idx = has_state_prep
    for op in tape.operations[has_state_prep:]:
        jacobian = [apply_operation(op, jac) for jac in jacobian]
        if op.num_params == 1:
            if param_idx in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                jacobian.append(apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), state))
            param_idx += 1
        state = apply_operation(op, state)
    return tuple((jac.flatten() for jac in jacobian))