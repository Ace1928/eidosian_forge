from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
class fix_discrete_solution_in_subproblem(object):

    def __init__(self, true_disjuncts, boolean_var_values, integer_var_values, subprob_util_block, config, solver):
        self.True_disjuncts = true_disjuncts
        self.boolean_var_values = boolean_var_values
        self.discrete_var_values = integer_var_values
        self.subprob_util_block = subprob_util_block
        self.config = config

    def __enter__(self):
        fixed = []
        for block in self.subprob_util_block.disjunct_list:
            if block in self.True_disjuncts:
                block.binary_indicator_var.fix(1)
                fixed.append(block.name)
            else:
                block.deactivate()
                block.binary_indicator_var.fix(0)
        self.config.logger.debug("Fixed the following Disjuncts to 'True': %s" % ', '.join(fixed))
        fixed_bools = []
        for subprob_bool, val in zip(self.subprob_util_block.non_indicator_boolean_variable_list, self.boolean_var_values):
            subprob_binary = subprob_bool.get_associated_binary()
            if val:
                subprob_binary.fix(1)
            else:
                subprob_binary.fix(0)
            fixed_bools.append('%s = %s' % (subprob_bool.name, val))
        self.config.logger.debug('Fixed the following Boolean variables: %s' % ', '.join(fixed_bools))
        if self.config.force_subproblem_nlp:
            fixed_discrete = []
            for subprob_var, val in zip(self.subprob_util_block.discrete_variable_list, self.discrete_var_values):
                fix_discrete_var(subprob_var, val, self.config)
                fixed_discrete.append('%s = %s' % (subprob_var.name, val))
            self.config.logger.debug('Fixed the following integer variables: %s' % ', '.join(fixed_discrete))
        self.config.subproblem_initialization_method(self.True_disjuncts, self.boolean_var_values, self.discrete_var_values, self.subprob_util_block)
        return self

    def __exit__(self, type, value, traceback):
        for block in self.subprob_util_block.disjunct_list:
            block.activate()
            block.binary_indicator_var.unfix()
        for bool_var in self.subprob_util_block.non_indicator_boolean_variable_list:
            bool_var.get_associated_binary().unfix()
        if self.config.force_subproblem_nlp:
            for subprob_var in self.subprob_util_block.discrete_variable_list:
                subprob_var.fixed = False