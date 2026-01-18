from itertools import product
from pyomo.common.collections import ComponentSet
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.nlp_initialization import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.solve_subproblem import solve_subproblem
from pyomo.contrib.gdpopt.util import (
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc
from pyomo.opt.base import SolverFactory
def _log_current_state(self, logger, subproblem_type, primal_improved=False):
    star = '*' if primal_improved else ''
    logger.info(self.log_formatter.format('{}/{}'.format(self.iteration, self.num_discrete_solns), subproblem_type, self.LB, self.UB, self.relative_gap(), get_main_elapsed_time(self.timing), star))