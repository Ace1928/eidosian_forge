from cvxpy import problems
from cvxpy.error import ParameterError
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.reductions.reduction import Reduction
class EvalParams(Reduction):
    """Replaces symbolic parameters with their constant values."""

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        """Replace parameters with constant values.

        Parameters
        ----------
        problem : Problem
            The problem whose parameters should be evaluated.

        Returns
        -------
        Problem
            A new problem where the parameters have been converted to constants.

        Raises
        ------
        ParameterError
            If the ``problem`` has unspecified parameters (i.e., a parameter
            whose value is None).
        """
        if len(problem.objective.parameters()) > 0:
            obj_expr = replace_params_with_consts(problem.objective.expr)
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective
        constraints = []
        for c in problem.constraints:
            args = []
            for arg in c.args:
                args.append(replace_params_with_consts(arg))
            if all((id(new) == id(old) for new, old in zip(args, c.args))):
                constraints.append(c)
            else:
                data = c.get_data()
                if data is not None:
                    constraints.append(type(c)(*args + data))
                else:
                    constraints.append(type(c)(*args))
        return (problems.problem.Problem(objective, constraints), [])

    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.
        """
        return solution