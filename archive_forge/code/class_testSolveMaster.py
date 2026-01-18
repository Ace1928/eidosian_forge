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
class testSolveMaster(unittest.TestCase):

    @unittest.skipUnless(baron_available, 'Global NLP solver is not available.')
    def test_solve_master(self):
        working_model = m = ConcreteModel()
        m.x = Var(initialize=0.5, bounds=(0, 10))
        m.y = Var(initialize=1.0, bounds=(0, 5))
        m.z = Var(initialize=0, bounds=(None, None))
        m.p = Param(initialize=1, mutable=True)
        m.obj = Objective(expr=m.x)
        m.con = Constraint(expr=m.x + m.y + m.z <= 3)
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        model_data.iteration = 0
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0, 0].transfer_attributes_from(working_model.clone())
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = [master_data.master_model.scenarios[0, 0].x]
        master_data.master_model.scenarios[0, 0].util.decision_rule_vars = []
        master_data.master_model.scenarios[0, 0].util.second_stage_variables = []
        master_data.master_model.scenarios[0, 0].util.uncertain_params = [master_data.master_model.scenarios[0, 0].p]
        master_data.master_model.scenarios[0, 0].first_stage_objective = 0
        master_data.master_model.scenarios[0, 0].second_stage_objective = Expression(expr=master_data.master_model.scenarios[0, 0].x)
        master_data.master_model.scenarios[0, 0].util.dr_var_to_exponent_map = ComponentMap()
        master_data.iteration = 0
        master_data.timing = TimingData()
        box_set = BoxSet(bounds=[(0, 2)])
        solver = SolverFactory(global_solver)
        config = ConfigBlock()
        config.declare('backup_global_solvers', ConfigValue(default=[]))
        config.declare('backup_local_solvers', ConfigValue(default=[]))
        config.declare('solve_master_globally', ConfigValue(default=True))
        config.declare('global_solver', ConfigValue(default=solver))
        config.declare('tee', ConfigValue(default=False))
        config.declare('decision_rule_order', ConfigValue(default=1))
        config.declare('objective_focus', ConfigValue(default=ObjectiveType.worst_case))
        config.declare('second_stage_variables', ConfigValue(default=master_data.master_model.scenarios[0, 0].util.second_stage_variables))
        config.declare('subproblem_file_directory', ConfigValue(default=None))
        config.declare('time_limit', ConfigValue(default=None))
        config.declare('progress_logger', ConfigValue(default=logging.getLogger(__name__)))
        with time_code(master_data.timing, 'main', is_main_timer=True):
            master_soln = solve_master(master_data, config)
            self.assertEqual(master_soln.termination_condition, TerminationCondition.optimal, msg='Could not solve simple master problem with solve_master function.')