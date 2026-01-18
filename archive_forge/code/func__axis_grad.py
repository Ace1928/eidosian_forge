import abc
from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
def _axis_grad(self, values) -> Optional[List[sp.csc_matrix]]:
    """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        Takes axis into account.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
    if self.axis is None or self.args[0].ndim < 2:
        value = np.reshape(values[0].T, (self.args[0].size, 1))
        D = self._column_grad(value)
        if D is not None:
            D = sp.csc_matrix(D)
    else:
        m, n = self.args[0].shape
        if self.axis == 0:
            D = sp.csc_matrix((m * n, n), dtype=float)
            for i in range(n):
                value = values[0][:, i]
                d = self._column_grad(value).T
                if d is None:
                    return [None]
                else:
                    d = np.array(d).flatten()
                row = np.linspace(i * n, i * n + m - 1, m)
                col = np.ones(m) * i
                D = D + sp.csc_matrix((d, (row, col)), shape=(m * n, n))
        else:
            values = np.transpose(values[0])
            D = sp.csc_matrix((m * n, m), dtype=float)
            for i in range(m):
                value = values[:, i]
                d = self._column_grad(value).T
                if d is None:
                    return [None]
                row = np.linspace(i, i + (n - 1) * m, n)
                col = np.ones(n) * i
                D = D + sp.csc_matrix((np.array(d)[0], (row, col)), shape=(m * n, m))
    return [D]