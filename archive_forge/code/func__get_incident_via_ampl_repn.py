from contextlib import nullcontext
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.repn.plugins.nl_writer import AMPLRepn
from pyomo.contrib.incidence_analysis.config import (
def _get_incident_via_ampl_repn(expr, linear_only, visitor):
    var_map = visitor.var_map
    orig_activevisitor = AMPLRepn.ActiveVisitor
    AMPLRepn.ActiveVisitor = visitor
    try:
        repn = visitor.walk_expression((expr, None, 0, 1.0))
    finally:
        AMPLRepn.ActiveVisitor = orig_activevisitor
    nonlinear_var_ids = [] if repn.nonlinear is None else repn.nonlinear[1]
    nonlinear_var_id_set = set()
    unique_nonlinear_var_ids = []
    for v_id in nonlinear_var_ids:
        if v_id not in nonlinear_var_id_set:
            nonlinear_var_id_set.add(v_id)
            unique_nonlinear_var_ids.append(v_id)
    nonlinear_vars = [var_map[v_id] for v_id in unique_nonlinear_var_ids]
    linear_only_vars = [var_map[v_id] for v_id, coef in repn.linear.items() if coef != 0.0 and v_id not in nonlinear_var_id_set]
    if linear_only:
        return linear_only_vars
    else:
        variables = linear_only_vars + nonlinear_vars
        return variables