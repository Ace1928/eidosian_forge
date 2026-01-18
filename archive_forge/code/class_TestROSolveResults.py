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
class TestROSolveResults(unittest.TestCase):
    """
    Test PyROS solver results object.
    """

    def test_ro_solve_results_str(self):
        """
        Test string representation of RO solve results object.
        """
        res = ROSolveResults(config=SolverFactory('pyros').CONFIG(), iterations=4, final_objective_value=123.456789, time=300.34567, pyros_termination_condition=pyrosTerminationCondition.robust_optimal)
        ans = 'Termination stats:\n Iterations            : 4\n Solve time (wall s)   : 300.346\n Final objective value : 1.2346e+02\n Termination condition : pyrosTerminationCondition.robust_optimal'
        self.assertEqual(str(res), ans, msg='String representation of PyROS results object does not match expected value')

    def test_ro_solve_results_str_attrs_none(self):
        """
        Test string representation of PyROS solve results in event
        one of the printed attributes is of value `None`.
        This may occur at instantiation or, for example,
        whenever the PyROS solver confirms robust infeasibility through
        coefficient matching.
        """
        res = ROSolveResults(config=SolverFactory('pyros').CONFIG(), iterations=0, final_objective_value=None, time=300.34567, pyros_termination_condition=pyrosTerminationCondition.robust_optimal)
        ans = 'Termination stats:\n Iterations            : 0\n Solve time (wall s)   : 300.346\n Final objective value : None\n Termination condition : pyrosTerminationCondition.robust_optimal'
        self.assertEqual(str(res), ans, msg='String representation of PyROS results object does not match expected value')