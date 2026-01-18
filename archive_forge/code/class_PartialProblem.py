from typing import List, Optional, Tuple
import cvxpy.settings as s
import cvxpy.utilities as u
from cvxpy.atoms import sum, trace
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
class PartialProblem(Expression):
    """A partial optimization problem.

    Attributes
    ----------
    opt_vars : list
        The variables to optimize over.
    dont_opt_vars : list
        The variables to not optimize over.
    """

    def __init__(self, prob: Problem, opt_vars: List[Variable], dont_opt_vars: List[Variable], solver, **kwargs) -> None:
        self.opt_vars = opt_vars
        self.dont_opt_vars = dont_opt_vars
        self.solver = solver
        self.args = [prob]
        self._solve_kwargs = kwargs
        super(PartialProblem, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.opt_vars, self.dont_opt_vars, self.solver]

    def is_constant(self) -> bool:
        return len(self.args[0].variables()) == 0

    def is_convex(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dcp() and type(self.args[0].objective) == Minimize

    def is_concave(self) -> bool:
        """Is the expression concave?
        """
        return self.args[0].is_dcp() and type(self.args[0].objective) == Maximize

    def is_dpp(self, context: str='dcp') -> bool:
        """The expression is a disciplined parameterized expression.
        """
        if context.lower() in ['dcp', 'dgp']:
            return self.args[0].is_dpp(context)
        else:
            raise ValueError('Unsupported context', context)

    def is_log_log_convex(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dgp() and type(self.args[0].objective) == Minimize

    def is_log_log_concave(self) -> bool:
        """Is the expression convex?
        """
        return self.args[0].is_dgp() and type(self.args[0].objective) == Maximize

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return self.args[0].objective.args[0].is_nonneg()

    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?
        """
        return self.args[0].objective.args[0].is_nonpos()

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        return False

    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the (row, col) dimensions of the expression.
        """
        return tuple()

    def name(self) -> str:
        """Returns the string representation of the expression.
        """
        return f'PartialProblem({self.args[0]})'

    def variables(self) -> List[Variable]:
        """Returns the variables in the problem.
        """
        return self.args[0].variables()

    def parameters(self):
        """Returns the parameters in the problem.
        """
        return self.args[0].parameters()

    def constants(self) -> List[Constant]:
        """Returns the constants in the problem.
        """
        return self.args[0].constants()

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        if self.is_constant():
            return u.grad.constant_grad(self)
        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.dont_opt_vars:
            if var.value is None:
                return u.grad.error_grad(self)
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective, fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        if prob.status in s.SOLUTION_PRESENT:
            sign = self.is_convex() - self.is_concave()
            lagr = self.args[0].objective.args[0]
            for constr in self.args[0].constraints:
                lagr_multiplier = self.cast_to_const(sign * constr.dual_value)
                prod = lagr_multiplier.T @ constr.expr
                if prod.is_scalar():
                    lagr += sum(prod)
                else:
                    lagr += trace(prod)
            grad_map = lagr.grad
            result = {var: grad_map[var] for var in self.dont_opt_vars}
        else:
            result = u.grad.error_grad(self)
        for var in self.variables():
            var.value = old_vals[var.id]
        return result

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        obj_expr = self.args[0].objective.args[0]
        return self.args[0].constraints + obj_expr.domain

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.dont_opt_vars:
            if var.value is None:
                return None
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective, fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        for var in self.variables():
            var.value = old_vals[var.id]
        return prob._solution.opt_val

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Change the ids of all the opt_vars.

        Returns
        -------
            A tuple of (affine expression, [constraints]).
        """
        obj, constrs = self.args[0].objective.args[0].canonical_form
        for cons in self.args[0].constraints:
            constrs += cons.canonical_form[1]
        return (obj, constrs)