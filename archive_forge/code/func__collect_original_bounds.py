from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
def _collect_original_bounds(discrete_prob_util_block):
    original_bounds = ComponentMap()
    for v in discrete_prob_util_block.all_mip_variables:
        original_bounds[v] = (v.lb, v.ub)
    return original_bounds