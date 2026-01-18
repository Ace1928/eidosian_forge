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
def _decompose_pauli_x_based_no_control_values(op: Controlled):
    """Decomposes a PauliX-based operation"""
    if isinstance(op.base, qml.PauliX) and len(op.control_wires) == 1:
        return [qml.CNOT(wires=op.active_wires)]
    if isinstance(op.base, qml.PauliX) and len(op.control_wires) == 2:
        return qml.Toffoli.compute_decomposition(wires=op.active_wires)
    if isinstance(op.base, qml.CNOT) and len(op.control_wires) == 1:
        return qml.Toffoli.compute_decomposition(wires=op.active_wires)
    return qml.MultiControlledX.compute_decomposition(wires=op.active_wires, work_wires=op.work_wires)