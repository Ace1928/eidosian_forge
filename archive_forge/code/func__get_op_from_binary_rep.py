import functools
from operator import matmul
import numpy as np
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RX, RY, RZ, PhaseShift
def _get_op_from_binary_rep(binary_rep, theta, wires):
    if len(binary_rep) == 1:
        op = PhaseShift(theta, wires[0]) if int(binary_rep) else PauliX(wires[0]) @ PhaseShift(theta, wires[0]) @ PauliX(wires[0])
    else:
        base_op = PhaseShift(theta, wires[-1]) if int(binary_rep[-1]) else PauliX(wires[-1]) @ PhaseShift(theta, wires[-1]) @ PauliX(wires[-1])
        op = qml.ctrl(base_op, control=wires[:-1], control_values=[int(i) for i in binary_rep[:-1]])
    return op