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
class testTransformToStandardForm(unittest.TestCase):

    def test_transform_to_std_form(self):
        """Check that `pyros.util.transform_to_standard_form` works
        correctly for an example model. That is:
        - all Constraints with a finite `upper` or `lower` attribute
          are either equality constraints, or inequalities
          of the standard form `expression(vars) <= upper`;
        - every inequality Constraint for which the `upper` and `lower`
          attribute are identical is converted to an equality constraint;
        - every inequality Constraint with distinct finite `upper` and
          `lower` attributes is split into two standard form inequality
          Constraints.
        """
        m = ConcreteModel()
        m.p = Param(initialize=1, mutable=True)
        m.x = Var(initialize=0)
        m.y = Var(initialize=1)
        m.z = Var(initialize=1)
        m.c1 = Constraint(expr=m.x >= 1)
        m.c2 = Constraint(expr=-m.y <= 0)
        m.c3 = Constraint(rule=(None, m.x + m.y, None))
        m.c4 = Constraint(rule=(1, m.x + m.y, 2))
        m.c5 = Constraint(rule=(m.p, m.x, m.p))
        m.c6 = Constraint(rule=(1.0, m.z, 1.0))
        clist = ConstraintList()
        m.add_component('clist', clist)
        clist.add(m.y <= 0)
        clist.add(m.x >= 1)
        clist.add((0, m.x, 1))
        num_orig_cons = len([con for con in m.component_data_objects(Constraint, active=True, descend_into=True)])
        num_lbub_cons = len([con for con in m.component_data_objects(Constraint, active=True, descend_into=True) if con.lower is not None and con.upper is not None and (con.lower is not con.upper)])
        num_nobound_cons = len([con for con in m.component_data_objects(Constraint, active=True, descend_into=True) if con.lower is None and con.upper is None])
        transform_to_standard_form(m)
        cons = [con for con in m.component_data_objects(Constraint, active=True, descend_into=True)]
        for con in cons:
            has_lb_or_ub = not (con.lower is None and con.upper is None)
            if has_lb_or_ub and (not con.equality):
                self.assertTrue(con.lower is None, msg='Constraint %s not in standard form' % con.name)
                lb_is_ub = con.lower is con.upper
                self.assertFalse(lb_is_ub, msg='Constraint %s should be converted to equality' % con.name)
            if con is not m.c3:
                self.assertTrue(has_lb_or_ub, msg='Constraint %s should have a lower or upper bound' % con.name)
        self.assertEqual(len([con for con in m.component_data_objects(Constraint, active=True, descend_into=True)]), num_orig_cons + num_lbub_cons - num_nobound_cons, msg='Expected number of constraints after\n standardizing constraints not matched. Number of constraints after\n transformation should be (number constraints in original model) \n + (number of constraints with distinct finite lower and upper bounds).')

    def test_transform_does_not_alter_num_of_constraints(self):
        """
        Check that if model does not contain any constraints
        for which both the `lower` and `upper` attributes are
        distinct and not None, then number of constraints remains the same
        after constraint standardization.
        Standard form for the purpose of PyROS is all inequality constraints
        as `g(.)<=0`.
        """
        m = ConcreteModel()
        m.x = Var(initialize=1, bounds=(0, 1))
        m.y = Var(initialize=0, bounds=(None, 1))
        m.con1 = Constraint(expr=m.x >= 1 + m.y)
        m.con2 = Constraint(expr=m.x ** 2 + m.y ** 2 >= 9)
        original_num_constraints = len(list(m.component_data_objects(Constraint)))
        transform_to_standard_form(m)
        final_num_constraints = len(list(m.component_data_objects(Constraint)))
        self.assertEqual(original_num_constraints, final_num_constraints, msg='Transform to standard form function led to a different number of constraints than in the original model.')
        number_of_non_standard_form_inequalities = len(list((c for c in list(m.component_data_objects(Constraint)) if c.lower != None)))
        self.assertEqual(number_of_non_standard_form_inequalities, 0, msg='All inequality constraints were not transformed to standard form.')