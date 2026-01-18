from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _fourier_motzkin_elimination(self, constraints, vars_to_eliminate):
    """Performs FME on the constraint list in the argument
        (which is assumed to be all >= constraints and stored in the
        dictionary representation), projecting out each of the variables in
        vars_to_eliminate"""
    vars_that_appear = []
    vars_that_appear_set = ComponentSet()
    for cons in constraints:
        std_repn = cons['body']
        if not std_repn.is_linear():
            nonlinear_vars = ComponentSet((v for two_tuple in std_repn.quadratic_vars for v in two_tuple))
            nonlinear_vars.update((v for v in std_repn.nonlinear_vars))
            for var in nonlinear_vars:
                if var in vars_to_eliminate:
                    raise RuntimeError('Variable %s appears in a nonlinear constraint. The Fourier-Motzkin Elimination transformation can only be used to eliminate variables which only appear linearly.' % var.name)
        for var in std_repn.linear_vars:
            if var in vars_to_eliminate:
                if not var in vars_that_appear_set:
                    vars_that_appear.append(var)
                    vars_that_appear_set.add(var)
    total = len(vars_that_appear)
    iteration = 1
    while vars_that_appear:
        the_var = vars_that_appear.pop()
        logger.warning('Projecting out var %s of %s' % (iteration, total))
        if self.verbose:
            logger.info('Projecting out %s' % the_var.getname(fully_qualified=True))
            logger.info('New constraints are:')
        leq_list = []
        geq_list = []
        waiting_list = []
        coefs = []
        for cons in constraints:
            leaving_var_coef = cons['map'].get(the_var)
            if leaving_var_coef is None or leaving_var_coef == 0:
                waiting_list.append(cons)
                if self.verbose:
                    logger.info('\t%s <= %s' % (cons['lower'], cons['body'].to_expression()))
                continue
            if not self.do_integer_arithmetic:
                if leaving_var_coef < 0:
                    leq_list.append(self._nonneg_scalar_multiply_linear_constraint(cons, -1.0 / leaving_var_coef))
                else:
                    geq_list.append(self._nonneg_scalar_multiply_linear_constraint(cons, 1.0 / leaving_var_coef))
            else:
                coefs.append(self._as_integer(leaving_var_coef, self._get_noninteger_coef_error_message, (the_var.name, leaving_var_coef)))
        if self.do_integer_arithmetic and len(coefs) > 0:
            least_common_mult = lcm(coefs)
            for cons in constraints:
                leaving_var_coef = cons['map'].get(the_var)
                if leaving_var_coef is None or leaving_var_coef == 0:
                    continue
                to_lcm = least_common_mult // abs(int(leaving_var_coef))
                if leaving_var_coef < 0:
                    leq_list.append(self._nonneg_scalar_multiply_linear_constraint(cons, to_lcm))
                else:
                    geq_list.append(self._nonneg_scalar_multiply_linear_constraint(cons, to_lcm))
        constraints = waiting_list
        for leq in leq_list:
            for geq in geq_list:
                constraints.append(self._add_linear_constraints(leq, geq))
                if self.verbose:
                    cons = constraints[len(constraints) - 1]
                    logger.info('\t%s <= %s' % (cons['lower'], cons['body'].to_expression()))
        iteration += 1
    return constraints