from typing import Tuple
from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS as cone_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.qp2quad_form.canonicalizers import QUAD_CANON_METHODS as quad_canon_methods
class Dcp2Cone(Canonicalization):
    """Reduce DCP problems to a conic form.

    This reduction takes as input (minimization) DCP problems and converts
    them into problems with affine or quadratic objectives and conic
    constraints whose arguments are affine.
    """

    def __init__(self, problem=None, quad_obj: bool=False) -> None:
        super(Canonicalization, self).__init__(problem=problem)
        self.cone_canon_methods = cone_canon_methods
        self.quad_canon_methods = quad_canon_methods
        self.quad_obj = quad_obj

    def accepts(self, problem):
        """A problem is accepted if it is a minimization and is DCP.
        """
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        """Converts a DCP problem to a conic form.
        """
        if not self.accepts(problem):
            raise ValueError('Cannot reduce problem to cone program')
        inverse_data = InverseData(problem)
        canon_objective, canon_constraints = self.canonicalize_tree(problem.objective, True)
        for constraint in problem.constraints:
            canon_constr, aux_constr = self.canonicalize_tree(constraint, False)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})
        new_problem = problems.problem.Problem(canon_objective, canon_constraints)
        return (new_problem, inverse_data)

    def canonicalize_tree(self, expr, affine_above: bool) -> Tuple[Expression, list]:
        """Recursively canonicalize an Expression.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        if type(expr) == cvxtypes.partial_problem():
            canon_expr, constrs = self.canonicalize_tree(expr.args[0].objective.expr, False)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(constr, False)
                constrs += [canon_constr] + aux_constr
        else:
            affine_atom = type(expr) not in self.cone_canon_methods
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(arg, affine_atom and affine_above)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args, affine_above)
            constrs += c
        return (canon_expr, constrs)

    def canonicalize_expr(self, expr, args, affine_above: bool) -> Tuple[Expression, list]:
        """Canonicalize an expression, w.r.t. canonicalized arguments.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        args : The canonicalized arguments of expr.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        if isinstance(expr, Expression) and (expr.is_constant() and (not expr.parameters())):
            return (expr, [])
        if self.quad_obj and affine_above and (type(expr) in self.quad_canon_methods):
            if type(expr) == cvxtypes.power() and (not expr._quadratic_power()):
                return self.cone_canon_methods[type(expr)](expr, args)
            else:
                return self.quad_canon_methods[type(expr)](expr, args)
        if type(expr) in self.cone_canon_methods:
            return self.cone_canon_methods[type(expr)](expr, args)
        return (expr.copy(args), [])