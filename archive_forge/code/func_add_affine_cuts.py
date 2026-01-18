from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
def add_affine_cuts(target_model, config, timing):
    """Adds affine cuts using MCPP.

    Parameters
    ----------
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    """
    with time_code(timing, 'Affine cut generation'):
        m = target_model
        config.logger.debug('Adding affine cuts')
        counter = 0
        for constr in m.MindtPy_utils.nonlinear_constraint_list:
            vars_in_constr = list(EXPR.identify_variables(constr.body))
            if any((var.value is None for var in vars_in_constr)):
                continue
            try:
                mc_eqn = mc(constr.body)
            except MCPP_Error as e:
                config.logger.error(e, exc_info=True)
                config.logger.error('Skipping constraint %s due to MCPP error' % constr.name)
                continue
            ccSlope = mc_eqn.subcc()
            cvSlope = mc_eqn.subcv()
            ccStart = mc_eqn.concave()
            cvStart = mc_eqn.convex()
            concave_cut_valid = True
            convex_cut_valid = True
            for var in vars_in_constr:
                if not var.fixed:
                    if ccSlope[var] == float('nan') or ccSlope[var] == float('inf'):
                        concave_cut_valid = False
                    if cvSlope[var] == float('nan') or cvSlope[var] == float('inf'):
                        convex_cut_valid = False
            if not any(list(ccSlope.values())):
                concave_cut_valid = False
            if not any(list(cvSlope.values())):
                convex_cut_valid = False
            if ccStart == float('nan') or ccStart == float('inf'):
                concave_cut_valid = False
            if cvStart == float('nan') or cvStart == float('inf'):
                convex_cut_valid = False
            if not (concave_cut_valid or convex_cut_valid):
                continue
            ub_int = min(value(constr.upper), mc_eqn.upper()) if constr.has_ub() else mc_eqn.upper()
            lb_int = max(value(constr.lower), mc_eqn.lower()) if constr.has_lb() else mc_eqn.lower()
            aff_cuts = m.MindtPy_utils.cuts.aff_cuts
            if concave_cut_valid:
                concave_cut = sum((ccSlope[var] * (var - var.value) for var in vars_in_constr if not var.fixed)) + ccStart >= lb_int
                aff_cuts.add(expr=concave_cut)
                counter += 1
            if convex_cut_valid:
                convex_cut = sum((cvSlope[var] * (var - var.value) for var in vars_in_constr if not var.fixed)) + cvStart <= ub_int
                aff_cuts.add(expr=convex_cut)
                counter += 1
        config.logger.debug('Added %s affine cuts' % counter)