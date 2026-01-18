import warnings
import functools
from copy import copy
from functools import wraps
from inspect import signature
from typing import List
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane.compiler import compiler
from .symbolicop import SymbolicOp
from .controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
def _compute_matrix_from_base(self):
    base_matrix = self.base.matrix()
    interface = qmlmath.get_interface(base_matrix)
    num_target_states = 2 ** len(self.target_wires)
    num_control_states = 2 ** len(self.control_wires)
    total_matrix_size = num_control_states * num_target_states
    padding_left = self._control_int * num_target_states
    padding_right = total_matrix_size - padding_left - num_target_states
    left_pad = qmlmath.convert_like(qmlmath.cast_like(qmlmath.eye(padding_left, like=interface), 1j), base_matrix)
    right_pad = qmlmath.convert_like(qmlmath.cast_like(qmlmath.eye(padding_right, like=interface), 1j), base_matrix)
    shape = qml.math.shape(base_matrix)
    if len(shape) == 3:
        return qml.math.stack([qml.math.block_diag([left_pad, _U, right_pad]) for _U in base_matrix])
    return qmlmath.block_diag([left_pad, base_matrix, right_pad])