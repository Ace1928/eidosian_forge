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
class testModelIdentifyObjectives(unittest.TestCase):
    """
    This class contains tests for validating routines used to
    determine the first-stage and second-stage portions of a
    two-stage expression.
    """

    def test_identify_objectives(self):
        """
        Test first and second-stage objective identification
        for a simple two-stage model.
        """
        m = ConcreteModel()
        m.p = Param(range(4), initialize=1, mutable=True)
        m.q = Param(initialize=1)
        m.x = Var(range(4))
        m.z = Var()
        m.y = Var(initialize=2)
        m.obj = Objective(expr=(m.x[0] + m.y) * (sum((m.x[idx] * m.p[idx] for idx in range(3))) + m.q * m.z + m.x[0] * m.q) + sin(m.x[0] + m.q) + cos(m.x[2] + m.z))
        m.util = Block()
        m.util.first_stage_variables = list(m.x.values())
        m.util.second_stage_variables = [m.z]
        m.util.uncertain_params = [m.p[0], m.p[1]]
        identify_objective_functions(m, m.obj)
        fsv_set = ComponentSet(m.util.first_stage_variables)
        uncertain_param_set = ComponentSet(m.util.uncertain_params)
        fsv_in_obj = ComponentSet((var for var in identify_variables(m.obj) if var in fsv_set))
        ssv_in_obj = ComponentSet((var for var in identify_variables(m.obj) if var not in fsv_set))
        uncertain_params_in_obj = ComponentSet((param for param in identify_mutable_parameters(m.obj) if param in uncertain_param_set))
        fsv_in_first_stg_cost = ComponentSet((var for var in identify_variables(m.first_stage_objective) if var in fsv_set))
        ssv_in_first_stg_cost = ComponentSet((var for var in identify_variables(m.first_stage_objective) if var not in fsv_set))
        uncertain_params_in_first_stg_cost = ComponentSet((param for param in identify_mutable_parameters(m.first_stage_objective) if param in uncertain_param_set))
        fsv_in_second_stg_cost = ComponentSet((var for var in identify_variables(m.second_stage_objective) if var in fsv_set))
        ssv_in_second_stg_cost = ComponentSet((var for var in identify_variables(m.second_stage_objective) if var not in fsv_set))
        uncertain_params_in_second_stg_cost = ComponentSet((param for param in identify_mutable_parameters(m.second_stage_objective) if param in uncertain_param_set))
        self.assertTrue(fsv_in_first_stg_cost | fsv_in_second_stg_cost == fsv_in_obj, f'{{var.name for var in fsv_in_first_stg_cost | fsv_in_second_stg_cost}} is not {{var.name for var in fsv_in_obj}}')
        self.assertFalse(ssv_in_first_stg_cost, f'First-stage expression {str(m.first_stage_objective.expr)} consists of non first-stage variables {{var.name for var in fsv_in_second_stg_cost}}')
        self.assertTrue(ssv_in_second_stg_cost == ssv_in_obj, f'{[var.name for var in ssv_in_second_stg_cost]} is not{{var.name for var in ssv_in_obj}}')
        self.assertFalse(uncertain_params_in_first_stg_cost, f'First-stage expression {str(m.first_stage_objective.expr)} consists of uncertain params {{p.name for p in uncertain_params_in_first_stg_cost}}')
        self.assertTrue(uncertain_params_in_second_stg_cost == uncertain_params_in_obj, f'{{p.name for p in uncertain_params_in_second_stg_cost}} is not {{p.name for p in uncertain_params_in_obj}}')

    def test_identify_objectives_var_expr(self):
        """
        Test first and second-stage objective identification
        for an objective expression consisting only of a Var.
        """
        m = ConcreteModel()
        m.p = Param(range(4), initialize=1, mutable=True)
        m.q = Param(initialize=1)
        m.x = Var(range(4))
        m.obj = Objective(expr=m.x[1])
        m.util = Block()
        m.util.first_stage_variables = list(m.x.values())
        m.util.second_stage_variables = list()
        m.util.uncertain_params = list()
        identify_objective_functions(m, m.obj)
        fsv_in_second_stg_obj = list((v.name for v in identify_variables(m.second_stage_objective)))
        self.assertTrue(list(identify_variables(m.first_stage_objective)) == [m.x[1]])
        self.assertFalse(fsv_in_second_stg_obj, f'Second stage objective contains variable(s) {fsv_in_second_stg_obj}')