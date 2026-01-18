from typing import Tuple
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from cvxpy.atoms.atom import Atom
class sigma_max(Atom):
    """ Maximum singular value. """
    _allow_complex = True

    def __init__(self, A) -> None:
        super(sigma_max, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest singular value of A.
        """
        return LA.norm(values[0], 2)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        U, s, V = LA.svd(values[0])
        ds = np.zeros(len(s))
        ds[0] = 1
        D = U.dot(np.diag(ds)).dot(V)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False