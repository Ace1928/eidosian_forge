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
def _is_single_qubit_special_unitary(op):
    if not op.has_matrix or len(op.wires) != 1:
        return False
    mat = op.matrix()
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return qmlmath.allclose(det, 1)