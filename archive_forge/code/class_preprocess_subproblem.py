from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException, DeveloperError
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Constraint, TransformationFactory, Objective, Block
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc
class preprocess_subproblem(object):

    def __init__(self, util_block, config):
        self.util_block = util_block
        self.config = config
        self.not_infeas = True
        self.unfixed_vars = []
        self.original_bounds = ComponentMap()
        self.constraints_deactivated = []
        self.constraints_modified = {}

    def __enter__(self):
        """Applies preprocessing transformations to the model."""
        m = self.util_block.parent_block()
        for cons in m.component_data_objects(Constraint, active=True, descend_into=Block):
            for v in EXPR.identify_variables(cons.expr):
                if v not in self.original_bounds.keys():
                    self.original_bounds[v] = (v.lb, v.ub)
                    if not v.fixed:
                        self.unfixed_vars.append(v)
        try:
            fbbt(m, integer_tol=self.config.integer_tolerance, feasibility_tol=self.config.constraint_tolerance, max_iter=self.config.max_fbbt_iterations)
            xfrm = TransformationFactory
            xfrm('contrib.detect_fixed_vars').apply_to(m, tolerance=self.config.variable_tolerance)
            if not self.config.tighten_nlp_var_bounds:
                for v, (lb, ub) in self.original_bounds.items():
                    v.setlb(lb)
                    v.setub(ub)
            xfrm('contrib.remove_zero_terms').apply_to(m, constraints_modified=self.constraints_modified)
            xfrm('contrib.deactivate_trivial_constraints').apply_to(m, tolerance=self.config.constraint_tolerance, return_trivial=self.constraints_deactivated)
        except InfeasibleConstraintException as e:
            self.config.logger.debug('NLP subproblem determined to be infeasible during preprocessing. Message: %s' % e)
            self.not_infeas = False
        return self.not_infeas

    def __exit__(self, type, value, traceback):
        if not self.not_infeas or self.config.tighten_nlp_var_bounds:
            for v, (lb, ub) in self.original_bounds.items():
                v.setlb(lb)
                v.setub(ub)
        for disj in self.util_block.disjunct_list:
            disj.binary_indicator_var.setlb(0)
            disj.binary_indicator_var.setub(1)
        for bool_var in self.util_block.non_indicator_boolean_variable_list:
            bool_var.get_associated_binary().setlb(0)
            bool_var.get_associated_binary().setub(1)
        for cons in self.constraints_deactivated:
            cons.activate()
        for cons, (orig, modified) in self.constraints_modified.items():
            cons.set_value(orig)
        for v in self.unfixed_vars:
            v.unfix()