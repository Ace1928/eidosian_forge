from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
def _column_grad(self, value):
    """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray matrix or None.
        """
    rows = value.size
    D_null = sp.csc_matrix((rows, 1), dtype='float64')
    value = value.reshape((rows, 1))
    D_null += value > 0
    D_null -= value < 0
    return D_null