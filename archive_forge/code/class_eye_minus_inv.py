from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
class eye_minus_inv(Atom):
    """The unity resolvent of a positive matrix, :math:`(I - X)^{-1}`.

    For an elementwise positive matrix :math:`X`, this atom represents

    .. math::

        (I - X)^{-1},

    and it enforces the constraint that the spectral radius of :math:`X`
    is at most :math:`1`.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    """

    def __init__(self, X) -> None:
        super(eye_minus_inv, self).__init__(X)
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError('The argument to `eye_minus_inv` must be a square matrix, received ', X)
        self.args[0] = X

    def numeric(self, values):
        return np.linalg.inv(np.eye(self.args[0].shape[0]) - values[0])

    def name(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self.args[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
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

    def _grad(self, values) -> None:
        return None