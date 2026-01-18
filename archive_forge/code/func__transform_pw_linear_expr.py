import pyomo.common.dependencies.numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
from pyomo.core import Constraint, NonNegativeIntegers, Suffix, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
    transBlock = transformation_block.transformed_functions[len(transformation_block.transformed_functions)]
    dimension = pw_expr.nargs()
    transBlock.disjuncts = Disjunct(NonNegativeIntegers)
    substitute_var = transBlock.substitute_var = Var()
    pw_linear_func.map_transformation_var(pw_expr, substitute_var)
    substitute_var_lb = float('inf')
    substitute_var_ub = -float('inf')
    if dimension > 1:
        A = np.ones((dimension + 1, dimension + 1))
        b = np.zeros(dimension + 1)
        b[-1] = 1
    for simplex, linear_func in zip(pw_linear_func._simplices, pw_linear_func._linear_functions):
        disj = transBlock.disjuncts[len(transBlock.disjuncts)]
        if dimension == 1:
            disj.simplex_halfspaces = Constraint(expr=(pw_linear_func._points[simplex[0]][0], pw_expr.args[0], pw_linear_func._points[simplex[1]][0]))
        else:
            disj.simplex_halfspaces = Constraint(range(dimension + 1))
            extreme_pts = []
            for idx in simplex:
                extreme_pts.append(pw_linear_func._points[idx])
            chull = spatial.ConvexHull(extreme_pts)
            vars = pw_expr.args
            for i, eqn in enumerate(chull.equations):
                disj.simplex_halfspaces[i] = sum((eqn[j] * v for j, v in enumerate(vars))) + float(eqn[dimension]) <= 0
        linear_func_expr = linear_func(*pw_expr.args)
        disj.set_substitute = Constraint(expr=substitute_var == linear_func_expr)
        lb, ub = compute_bounds_on_expr(linear_func_expr)
        if lb is not None and lb < substitute_var_lb:
            substitute_var_lb = lb
        if ub is not None and ub > substitute_var_ub:
            substitute_var_ub = ub
    if substitute_var_lb < float('inf'):
        transBlock.substitute_var.setlb(substitute_var_lb)
    if substitute_var_ub > -float('inf'):
        transBlock.substitute_var.setub(substitute_var_ub)
    transBlock.pick_a_piece = Disjunction(expr=[d for d in transBlock.disjuncts.values()])
    return transBlock.substitute_var