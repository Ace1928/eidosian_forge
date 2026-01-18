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
def _solve_gdp(self, original_model, config):
    logger = config.logger
    add_constraint_list(self.original_util_block)
    discrete_problem_util_block, subproblem_util_block = _get_discrete_problem_and_subproblem(self, config)
    discrete = discrete_problem_util_block.parent_block()
    subproblem = subproblem_util_block.parent_block()
    original_obj = self._setup_augmented_penalty_objective(discrete_problem_util_block)
    self._log_header(logger)
    while not config.iterlim or self.iteration < config.iterlim:
        self.iteration += 1
        with time_code(self.timing, 'mip'):
            oa_obj = self._update_augmented_penalty_objective(discrete_problem_util_block, original_obj, config.OA_penalty_factor)
            mip_feasible = solve_MILP_discrete_problem(discrete_problem_util_block, self, config)
            self._update_bounds_after_discrete_problem_solve(mip_feasible, oa_obj, logger)
        if self.any_termination_criterion_met(config):
            break
        with time_code(self.timing, 'nlp'):
            self._fix_discrete_soln_solve_subproblem_and_add_cuts(discrete_problem_util_block, subproblem_util_block, config)
        with time_code(self.timing, 'integer cut generation'):
            add_no_good_cut(discrete_problem_util_block, config)
        if self.any_termination_criterion_met(config):
            break