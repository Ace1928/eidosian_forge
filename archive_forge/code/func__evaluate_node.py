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
def _evaluate_node(self, node_data, node_model, config):
    config.logger.info('Exploring node %s with LB %.10g UB %.10g and %s inactive disjunctions.' % (node_data.node_count, node_data.obj_lb, node_data.obj_ub, node_data.num_unbranched_disjunctions))
    new_lb, new_ub = self._solve_rnGDP_subproblem(node_model, config)
    new_node_data = node_data._replace(obj_lb=new_lb, obj_ub=new_ub, is_evaluated=True)
    return new_node_data