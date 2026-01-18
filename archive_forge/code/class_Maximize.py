import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.error import DCPError
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.utilities import scopes
class Maximize(Objective):
    """An optimization objective for maximization.

    Parameters
    ----------
    expr : Expression
        The expression to maximize. Must be a scalar.

    Raises
    ------
    ValueError
        If expr is not a scalar.
    """
    NAME = 'maximize'

    def __neg__(self) -> Minimize:
        return Minimize(-self.args[0])

    def __add__(self, other):
        if not isinstance(other, (Minimize, Maximize)):
            raise NotImplementedError()
        if type(other) is Maximize:
            return Maximize(self.args[0] + other.args[0])
        else:
            raise Exception('Problem does not follow DCP rules.')

    def canonicalize(self):
        """Negates the target expression's objective.
        """
        obj, constraints = self.args[0].canonical_form
        return (lu.neg_expr(obj), constraints)

    def is_dcp(self, dpp: bool=False) -> bool:
        """The objective must be concave.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_concave()
        return self.args[0].is_concave()

    def is_dgp(self, dpp: bool=False) -> bool:
        """The objective must be log-log concave.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_log_log_concave()
        return self.args[0].is_log_log_concave()

    def is_dpp(self, context='dcp') -> bool:
        with scopes.dpp_scope():
            if context.lower() == 'dcp':
                return self.is_dcp(dpp=True)
            elif context.lower() == 'dgp':
                return self.is_dgp(dpp=True)
            else:
                raise ValueError('Unsupported context ', context)

    def is_dqcp(self) -> bool:
        """The objective must be quasiconcave.
        """
        return self.args[0].is_quasiconcave()

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return -result