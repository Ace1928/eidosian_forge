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
Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          (0.5) [Y0]
        + (1.0) [Z0 X1]

        