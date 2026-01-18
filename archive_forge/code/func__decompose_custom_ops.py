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
def _decompose_custom_ops(op: Controlled) -> List['operation.Operator']:
    """Custom handling for decomposing a controlled operation"""
    pauli_x_based_ctrl_ops = _get_pauli_x_based_ops()
    ops_with_custom_ctrl_ops = _get_special_ops()
    custom_key = (type(op.base), len(op.control_wires))
    if custom_key in ops_with_custom_ctrl_ops:
        custom_op_cls = ops_with_custom_ctrl_ops[custom_key]
        return custom_op_cls.compute_decomposition(*op.data, op.active_wires)
    if isinstance(op.base, pauli_x_based_ctrl_ops):
        return _decompose_pauli_x_based_no_control_values(op)
    if isinstance(op.base, qml.PhaseShift):
        base_decomp = qml.ControlledPhaseShift.compute_decomposition(*op.data, op.wires[-2:])
        return [ctrl(new_op, op.control_wires[:-1], work_wires=op.work_wires) for new_op in base_decomp]
    if len(op.control_wires) == 1 and hasattr(op.base, '_controlled'):
        result = op.base._controlled(op.control_wires[0])
        if type(result) != type(op):
            return [result]
        qml.QueuingManager.remove(result)
    return None