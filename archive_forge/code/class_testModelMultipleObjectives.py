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
@unittest.skipUnless(scip_available, 'Global NLP solver is not available.')
class testModelMultipleObjectives(unittest.TestCase):
    """
    This class contains tests for models with multiple
    Objective attributes.
    """

    def test_multiple_objs(self):
        """Test bypassing of global separation solve calls."""
        m = ConcreteModel()
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.u = Param(initialize=1.125, mutable=True)
        m.con1 = Constraint(expr=m.x1 * m.u ** 0.5 - m.x2 * m.u <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
        m.obj2 = Objective(expr=m.obj.expr / 2)
        m.b = Block()
        m.b.obj = Objective(expr=m.obj.expr / 2)
        interval = BoxSet(bounds=[(0.25, 2)])
        pyros_solver = SolverFactory('pyros')
        local_subsolver = SolverFactory('ipopt')
        global_subsolver = SolverFactory('scip')
        solve_kwargs = dict(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u], uncertainty_set=interval, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True, 'decision_rule_order': 0})
        with self.assertRaisesRegex(ValueError, 'Expected model with exactly 1 active objective.*has 3'):
            pyros_solver.solve(**solve_kwargs)
        m.b.obj.deactivate()
        with self.assertRaisesRegex(ValueError, 'Expected model with exactly 1 active objective.*has 2'):
            pyros_solver.solve(**solve_kwargs)
        m.obj2.deactivate()
        res = pyros_solver.solve(**solve_kwargs)
        self.assertIs(res.pyros_termination_condition, pyrosTerminationCondition.robust_optimal)
        self.assertEqual(len(list(m.component_data_objects(Objective, active=True))), 1)
        self.assertTrue(m.obj.active)
        m.obj_max = Objective(expr=-m.obj.expr, sense=pyo_max)
        m.obj.deactivate()
        max_obj_res = pyros_solver.solve(**solve_kwargs)
        self.assertEqual(len(list(m.component_data_objects(Objective, active=True))), 1)
        self.assertTrue(m.obj_max.active)
        self.assertTrue(math.isclose(res.final_objective_value, -max_obj_res.final_objective_value, abs_tol=0.0002), msg=f'Robust optimal objective value {res.final_objective_value} for problem with minimization objective not close to negative of value {max_obj_res.final_objective_value} of equivalent maximization objective.')