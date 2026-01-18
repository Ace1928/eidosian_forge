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
def _update_augmented_penalty_objective(self, discrete_problem_util_block, discrete_objective, OA_penalty_factor):
    m = discrete_problem_util_block.parent_block()
    sign_adjust = 1 if discrete_objective.sense == minimize else -1
    OA_penalty_expr = sign_adjust * OA_penalty_factor * sum((v for v in m.component_data_objects(ctype=Var, descend_into=(Block, Disjunct)) if v.parent_component().local_name == 'GDPopt_OA_slacks'))
    discrete_problem_util_block.oa_obj.expr = discrete_objective.expr + OA_penalty_expr
    return discrete_problem_util_block.oa_obj.expr