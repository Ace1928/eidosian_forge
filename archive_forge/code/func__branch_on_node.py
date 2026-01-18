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
def _branch_on_node(self, node_data, node_model, config):
    node_utils = node_model.component(self.original_util_block.name)
    disjunction_to_branch_idx = node_data.unbranched_disjunction_indices[0]
    disjunction_to_branch = node_utils.disjunction_list[disjunction_to_branch_idx]
    num_unfixed_disjuncts = len(node_utils.disjunction_to_unfixed_disjuncts[disjunction_to_branch])
    config.logger.info('Branching on disjunction %s' % disjunction_to_branch.name)
    node_count = self.created_nodes
    newly_created_nodes = 0
    for disjunct_index_to_fix_True in range(num_unfixed_disjuncts):
        child_model = node_model.clone()
        child_utils = child_model.component(node_utils.name)
        child_disjunction_to_branch = child_utils.disjunction_list[disjunction_to_branch_idx]
        child_unfixed_disjuncts = child_utils.disjunction_to_unfixed_disjuncts[child_disjunction_to_branch]
        for idx, child_disjunct in enumerate(child_unfixed_disjuncts):
            if idx == disjunct_index_to_fix_True:
                child_disjunct.indicator_var.fix(True)
            else:
                child_disjunct.deactivate()
        if not child_disjunction_to_branch.xor:
            raise NotImplementedError('We still need to add support for non-XOR disjunctions.')
        fixed_True_disjunct = child_unfixed_disjuncts[disjunct_index_to_fix_True]
        for constr in child_utils.disjunct_to_nonlinear_constraints.get(fixed_True_disjunct, ()):
            constr.activate()
            child_model.BigM[constr] = 1
        del child_utils.disjunction_to_unfixed_disjuncts[child_disjunction_to_branch]
        for child_disjunct in child_unfixed_disjuncts:
            child_utils.disjunct_to_nonlinear_constraints.pop(child_disjunct, None)
        newly_created_nodes += 1
        child_node_data = node_data._replace(is_screened=False, is_evaluated=False, num_unbranched_disjunctions=node_data.num_unbranched_disjunctions - 1, node_count=node_count + newly_created_nodes, unbranched_disjunction_indices=node_data.unbranched_disjunction_indices[1:], obj_ub=float('inf'))
        heappush(self.bb_queue, (child_node_data, child_model))
    self.created_nodes += newly_created_nodes
    config.logger.info('Added %s new nodes with %s relaxed disjunctions to the heap. Size now %s.' % (num_unfixed_disjuncts, node_data.num_unbranched_disjunctions - 1, len(self.bb_queue)))