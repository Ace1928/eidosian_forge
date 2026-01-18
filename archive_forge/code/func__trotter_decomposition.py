from typing import List
from warnings import warn
import numpy as np
from scipy.sparse.linalg import expm as sparse_expm
import pennylane as qml
from pennylane import math
from pennylane.math import expand_matrix
from pennylane.operation import (
from pennylane.ops.qubit import Hamiltonian
from pennylane.wires import Wires
from .sprod import SProd
from .sum import Sum
from .symbolicop import ScalarSymbolicOp
def _trotter_decomposition(self, ops: List[Operator], coeffs: List[complex]):
    """Uses the Suzuki-Trotter approximation to decompose the exponential of the linear
        combination of ``coeffs`` and ``ops``.

        Args:
            ops (List[Operator]): list of operators of the linear combination
            coeffs (List[complex]): list of coefficients of the linear combination

        Raises:
            ValueError: if the Trotter number (``num_steps``) is not defined
            DecompositionUndefinedError: if the linear combination contains operators that are not
                Pauli words

        Returns:
            List[Operator]: a list of operators containing the decomposition
        """
    op_list = []
    for c, op in zip(coeffs, ops):
        c /= self.num_steps
        if isinstance(op, SProd):
            c *= op.scalar
            op = op.base
        op_list.extend(self._recursive_decomposition(op, c))
    return op_list * self.num_steps