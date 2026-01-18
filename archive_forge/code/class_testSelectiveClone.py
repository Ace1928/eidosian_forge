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
class testSelectiveClone(unittest.TestCase):
    """
    Testing for the selective_clone function. This function takes as input a Pyomo model object
    and a list of variables objects "first_stage_vars" in that Pyomo model which should *not* be cloned.
    It returns a clone of the original Pyomo model object wherein the "first_stage_vars" members are unchanged,
    i.e. all cloned model expressions still reference the "first_stage_vars" of the original model object.
    """

    def test_cloning_negative_case(self):
        """
        Testing correct behavior if incorrect first_stage_vars list object is passed to selective_clone
        """
        m = ConcreteModel()
        m.x = Var(initialize=2)
        m.y = Var(initialize=2)
        m.p = Param(initialize=1)
        m.con = Constraint(expr=m.x * m.p + m.y <= 0)
        n = ConcreteModel()
        n.x = Var()
        m.first_stage_vars = [n.x]
        cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)
        self.assertNotEqual(id(m.first_stage_vars), id(cloned_model.first_stage_vars), msg='First stage variables should not be equal.')

    def test_cloning_positive_case(self):
        """
        Testing if selective_clone works correctly for correct first_stage_var object definition.
        """
        m = ConcreteModel()
        m.x = Var(initialize=2)
        m.y = Var(initialize=2)
        m.p = Param(initialize=1)
        m.con = Constraint(expr=m.x * m.p + m.y <= 0)
        m.first_stage_vars = [m.x]
        cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)
        self.assertEqual(id(m.x), id(cloned_model.x), msg='First stage variables should be equal.')
        self.assertNotEqual(id(m.y), id(cloned_model.y), msg='Non-first-stage variables should not be equal.')
        self.assertNotEqual(id(m.p), id(cloned_model.p), msg='Params should not be equal.')
        self.assertNotEqual(id(m.con), id(cloned_model.con), msg='Constraint objects should not be equal.')