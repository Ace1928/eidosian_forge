from collections import namedtuple
from math import copysign
from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import (
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct
from pyomo.opt.base import SolverFactory
from pyomo.repn import generate_standard_repn
def _add_cuts_to_discrete_problem(self, subproblem_util_block, discrete_problem_util_block, objective_sense, config, timing):
    """Add outer approximation cuts to the linear GDP model."""
    m = discrete_problem_util_block.parent_block()
    nlp = subproblem_util_block.parent_block()
    sign_adjust = -1 if objective_sense == minimize else 1
    if hasattr(discrete_problem_util_block, 'oa_cut_blocks'):
        oa_cut_blocks = discrete_problem_util_block.oa_cut_blocks
    else:
        oa_cut_blocks = discrete_problem_util_block.oa_cut_blocks = dict()
    for discrete_var, subprob_var in zip(discrete_problem_util_block.algebraic_variable_list, subproblem_util_block.algebraic_variable_list):
        val = subprob_var.value
        if val is not None and (not discrete_var.fixed):
            discrete_var.set_value(val, skip_validation=True)
    config.logger.debug('Adding OA cuts.')
    counter = 0
    if not hasattr(discrete_problem_util_block, 'jacobians'):
        discrete_problem_util_block.jacobians = ComponentMap()
    for constr, subprob_constr in zip(discrete_problem_util_block.constraint_list, subproblem_util_block.constraint_list):
        dual_value = nlp.dual.get(subprob_constr, None)
        if dual_value is None or generate_standard_repn(constr.body).is_linear():
            continue
        parent_block = constr.parent_block()
        ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
        config.logger.debug('Ignore_set %s' % ignore_set)
        if ignore_set and (constr in ignore_set or constr.parent_component() in ignore_set):
            config.logger.debug('OA cut addition for %s skipped because it is in the ignore set.' % constr.name)
            continue
        config.logger.debug('Adding OA cut for %s with dual value %s' % (constr.name, dual_value))
        jacobian = discrete_problem_util_block.jacobians.get(constr, None)
        if jacobian is None:
            constr_vars = list(identify_variables(constr.body, include_fixed=False))
            if len(constr_vars) >= MAX_SYMBOLIC_DERIV_SIZE:
                mode = differentiate.Modes.reverse_numeric
            else:
                mode = differentiate.Modes.sympy
            try:
                jac_list = differentiate(constr.body, wrt_list=constr_vars, mode=mode)
                jac_map = ComponentMap(zip(constr_vars, jac_list))
            except:
                if mode is differentiate.Modes.reverse_numeric:
                    raise
                mode = differentiate.Modes.reverse_numeric
                jac_map = ComponentMap()
            jacobian = JacInfo(mode=mode, vars=constr_vars, jac=jac_map)
            discrete_problem_util_block.jacobians[constr] = jacobian
        if not jacobian.jac:
            jac_list = differentiate(constr.body, wrt_list=jacobian.vars, mode=jacobian.mode)
            jacobian.jac.update(zip(jacobian.vars, jac_list))
        oa_utils = oa_cut_blocks.get(parent_block)
        if oa_utils is None:
            nm = unique_component_name(parent_block, 'GDPopt_OA_cuts')
            oa_utils = Block(doc='Block holding outer approximation cuts and associated data.')
            parent_block.add_component(nm, oa_utils)
            oa_cut_blocks[parent_block] = oa_utils
            oa_utils.cuts = Constraint(NonNegativeIntegers)
        discrete_prob_oa_utils = discrete_problem_util_block.component('GDPopt_OA_slacks')
        if discrete_prob_oa_utils is None:
            discrete_prob_oa_utils = discrete_problem_util_block.GDPopt_OA_slacks = Block(doc='Block holding outer approximation slacks for the whole model (so that the writers can find them).')
            discrete_prob_oa_utils.slacks = VarList(bounds=(0, config.max_slack), domain=NonNegativeReals, initialize=0)
        oa_cuts = oa_utils.cuts
        slack_var = discrete_prob_oa_utils.slacks.add()
        rhs = value(constr.lower) if constr.has_lb() else value(constr.upper)
        try:
            new_oa_cut = copysign(1, sign_adjust * dual_value) * (value(constr.body) - rhs + sum((value(jac) * (var - value(var)) for var, jac in jacobian.jac.items()))) - slack_var <= 0
            assert new_oa_cut.polynomial_degree() in (1, 0)
            idx = len(oa_cuts)
            oa_cuts[idx] = new_oa_cut
            _add_bigm_constraint_to_transformed_model(m, oa_cuts[idx], oa_cuts)
            config.logger.debug('Cut expression: %s' % new_oa_cut)
            counter += 1
        except ZeroDivisionError:
            config.logger.warning('Zero division occurred attempting to generate OA cut for constraint %s.\nSkipping OA cut generation for this constraint.' % (constr.name,))
        if jacobian.mode is differentiate.Modes.reverse_numeric:
            jacobian.jac.clear()
    config.logger.debug('Added %s OA cuts' % counter)