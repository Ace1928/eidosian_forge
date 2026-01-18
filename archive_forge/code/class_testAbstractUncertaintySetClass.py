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
class testAbstractUncertaintySetClass(unittest.TestCase):
    """
    The UncertaintySet class has an abstract base class implementing set_as_constraint method, as well as a couple
    basic uncertainty sets (ellipsoidal, polyhedral). The set_as_constraint method must return a Constraint object
    which references the Param objects from the uncertain_params list in the original model object.
    """

    def test_uncertainty_set_with_correct_params(self):
        """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = m.uncertain_params
        _set = myUncertaintySet()
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list((v for v in m.uncertain_param_vars if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))))
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars], msg='Uncertain param Var objects used to construct uncertainty set constraint mustbe the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the UncertaintySet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        _set = myUncertaintySet()
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_params)
        variables_in_constr = list((v for v in m.uncertain_params if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr.expr))))
        self.assertEqual(len(variables_in_constr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentiallyvariable expression.')