import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
def _controlled(self, wire):
    return qml.Toffoli(wires=wire + self.wires)