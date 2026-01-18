from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.atoms.elementwise.elementwise import Elementwise
Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        