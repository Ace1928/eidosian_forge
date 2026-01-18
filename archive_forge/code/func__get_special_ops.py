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
@functools.lru_cache()
def _get_special_ops():
    """Gets a list of special operations with custom controlled versions.

    This is placed inside a function to avoid circular imports.

    """
    ops_with_custom_ctrl_ops = {(qml.PauliZ, 1): qml.CZ, (qml.PauliZ, 2): qml.CCZ, (qml.PauliY, 1): qml.CY, (qml.CZ, 1): qml.CCZ, (qml.SWAP, 1): qml.CSWAP, (qml.Hadamard, 1): qml.CH, (qml.RX, 1): qml.CRX, (qml.RY, 1): qml.CRY, (qml.RZ, 1): qml.CRZ, (qml.Rot, 1): qml.CRot, (qml.PhaseShift, 1): qml.ControlledPhaseShift}
    return ops_with_custom_ctrl_ops