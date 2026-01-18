from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.atom import Atom
class one_minus_pos(Atom):
    """The difference :math:`1 - x` with domain `\\{x : 0 < x < 1\\}`.

    This atom is log-log concave.

    Parameters
    ----------
    x : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    """

    def __init__(self, x) -> None:
        super(one_minus_pos, self).__init__(x)
        self.args[0] = x
        self._ones = np.ones(self.args[0].shape)

    def numeric(self, values):
        return self._ones - values[0]

    def _grad(self, values):
        del values
        return sp.csc_matrix(-1.0 * self._ones)

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
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return True