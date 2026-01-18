import warnings
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities import scopes
class Inequality(Constraint):
    """A constraint of the form :math:`x \\leq y`.

    Dual variables to these constraints are always nonnegative.
    A constraint of this type affects the Lagrangian :math:`L` of a
    minimization problem by

        :math:`L += (x - y)^{T}(\\texttt{con.dual\\_value})`.

    The preferred way of creating one of these constraints is via
    operator overloading. The expression ``x <= y`` evaluates to
    ``Inequality(x, y)``, and the expression ``x >= y`` evaluates
    to ``Inequality(y, x)``.

    Parameters
    ----------
    lhs : Expression
        The expression to be upper-bounded by rhs
    rhs : Expression
        The expression to be lower-bounded by lhs
    constr_id : int
        A unique id for the constraint.
    """

    def __init__(self, lhs, rhs, constr_id=None) -> None:
        self._expr = lhs - rhs
        if self._expr.is_complex():
            raise ValueError('Inequality constraints cannot be complex.')
        super(Inequality, self).__init__([lhs, rhs], constr_id)

    def _construct_dual_variables(self, args) -> None:
        super(Inequality, self)._construct_dual_variables([self._expr])

    @property
    def expr(self):
        return self._expr

    def name(self) -> str:
        return '%s <= %s' % (self.args[0], self.args[1])

    @property
    def shape(self):
        """int : The shape of the constrained expression."""
        return self.expr.shape

    @property
    def size(self):
        """int : The size of the constrained expression."""
        return self.expr.size

    def is_dcp(self, dpp: bool=False) -> bool:
        """A non-positive constraint is DCP if its argument is convex."""
        if dpp:
            with scopes.dpp_scope():
                return self.expr.is_convex()
        return self.expr.is_convex()

    def is_dgp(self, dpp: bool=False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_log_log_convex() and self.args[1].is_log_log_concave()
        return self.args[0].is_log_log_convex() and self.args[1].is_log_log_concave()

    def is_dpp(self, context='dcp') -> bool:
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    def is_dqcp(self) -> bool:
        return self.is_dcp() or (self.args[0].is_quasiconvex() and self.args[1].is_constant()) or (self.args[0].is_constant() and self.args[1].is_quasiconcave())

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        ---------
        NumPy.ndarray
        """
        if self.expr.value is None:
            return None
        return np.maximum(self.expr.value, 0)