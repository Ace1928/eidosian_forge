from collections import namedtuple
from heapq import heappush, heappop
import traceback
from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, Constraint, TransformationFactory
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc
def _solve_rnGDP_subproblem(self, model, config):
    subproblem = TransformationFactory('gdp.bigm').create_using(model)
    obj_sense_correction = self.objective_sense != minimize
    model_utils = model.component(self.original_util_block.name)
    subprob_utils = subproblem.component(self.original_util_block.name)
    try:
        with SuppressInfeasibleWarning():
            try:
                fbbt(subproblem, integer_tol=config.integer_tolerance)
            except InfeasibleConstraintException:
                copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
                return (float('inf'), float('inf'))
            minlp_args = dict(config.minlp_solver_args)
            if config.time_limit is not None and config.minlp_solver == 'gams':
                elapsed = get_main_elapsed_time(self.timing)
                remaining = max(config.time_limit - elapsed, 1)
                minlp_args['add_options'] = minlp_args.get('add_options', [])
                minlp_args['add_options'].append('option reslim=%s;' % remaining)
            result = SolverFactory(config.minlp_solver).solve(subproblem, **minlp_args)
    except RuntimeError as e:
        config.logger.warning('Solver encountered RuntimeError. Treating as infeasible. Msg: %s\n%s' % (str(e), traceback.format_exc()))
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('inf'), float('inf'))
    term_cond = result.solver.termination_condition
    if term_cond == tc.optimal:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config)
        return (lb, ub)
    elif term_cond == tc.locallyOptimal or term_cond == tc.feasible:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config)
        return (lb, ub)
    elif term_cond == tc.unbounded:
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('-inf'), float('-inf'))
    elif term_cond == tc.infeasible:
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('inf'), float('inf'))
    else:
        config.logger.warning('Unknown termination condition of %s. Treating as infeasible.' % term_cond)
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('inf'), float('inf'))