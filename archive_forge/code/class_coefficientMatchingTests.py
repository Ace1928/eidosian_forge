import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class coefficientMatchingTests(unittest.TestCase):

    def test_coefficient_matching_correct_num_constraints_added(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)
        m.con = Constraint(expr=m.u ** 0.5 * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr=m.u ** 2 * (m.x2 - 1) + m.u * (m.x1 ** 3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) == 0)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
        config = Block()
        config.uncertainty_set = Block()
        config.uncertainty_set.parameter_bounds = [(0.25, 2)]
        m.util = Block()
        m.util.first_stage_variables = [m.x1, m.x2]
        m.util.second_stage_variables = []
        m.util.uncertain_params = [m.u]
        config.decision_rule_order = 0
        m.util.h_x_q_constraints = ComponentSet()
        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)
        self.assertEqual(coeff_matching_success, True, msg='Coefficient matching was unsuccessful.')
        self.assertEqual(robust_infeasible, False, msg='Coefficient matching detected a robust infeasible constraint (1 == 0).')
        self.assertEqual(len(m.coefficient_matching_constraints), 2, msg='Coefficient matching produced incorrect number of h(x,q)=0 constraints.')
        config.decision_rule_order = 1
        model_data = Block()
        model_data.working_model = m
        m.util.first_stage_variables = [m.x1]
        m.util.second_stage_variables = [m.x2]
        add_decision_rule_variables(model_data=model_data, config=config)
        add_decision_rule_constraints(model_data=model_data, config=config)
        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)
        self.assertEqual(coeff_matching_success, False, msg='Coefficient matching should have been unsuccessful for higher order polynomial expressions.')
        self.assertEqual(robust_infeasible, False, msg='Coefficient matching is not successful, but should not be proven robust infeasible.')

    def test_coefficient_matching_robust_infeasible_proof(self):
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.u = Param(initialize=1.125, mutable=True)
        m.con = Constraint(expr=m.u ** 0.5 * m.x1 - m.u * m.x2 <= 2)
        m.eq_con = Constraint(expr=m.u * (m.x1 ** 3 + 0.5) - 5 * m.u * m.x1 * m.x2 + m.u * (m.x1 + 2) + m.u ** 2 == 0)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
        config = Block()
        config.uncertainty_set = Block()
        config.uncertainty_set.parameter_bounds = [(0.25, 2)]
        m.util = Block()
        m.util.first_stage_variables = [m.x1, m.x2]
        m.util.second_stage_variables = []
        m.util.uncertain_params = [m.u]
        config.decision_rule_order = 0
        m.util.h_x_q_constraints = ComponentSet()
        coeff_matching_success, robust_infeasible = coefficient_matching(m, m.eq_con, [m.u], config)
        self.assertEqual(coeff_matching_success, False, msg='Coefficient matching should have been unsuccessful.')
        self.assertEqual(robust_infeasible, True, msg='Coefficient matching should be proven robust infeasible.')