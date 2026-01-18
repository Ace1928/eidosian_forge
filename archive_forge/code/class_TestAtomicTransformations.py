import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
@unittest.skipUnless(sympy_available, 'Sympy not available')
class TestAtomicTransformations(unittest.TestCase):

    def test_implies(self):
        m = ConcreteModel()
        m.x = BooleanVar()
        m.y = BooleanVar()
        m.p = LogicalConstraint(expr=m.x.implies(m.y))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(1, 1 - m.x.get_associated_binary() + m.y.get_associated_binary(), None)], m.logic_to_linear.transformed_constraints)

    def test_literal(self):
        m = ConcreteModel()
        m.Y = BooleanVar()
        m.p = LogicalConstraint(expr=m.Y)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(1, m.Y.get_associated_binary(), 1)], m.logic_to_linear.transformed_constraints)

    def test_constant_True(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError, "LogicalConstraint 'p' is always True."):
            m.p = LogicalConstraint(expr=True)
            TransformationFactory('core.logical_to_linear').apply_to(m)
        self.assertIsNone(m.component('logic_to_linear'))

    def test_nothing_to_do(self):
        m = ConcreteModel()
        m.p = LogicalConstraint()
        TransformationFactory('core.logical_to_linear').apply_to(m)
        self.assertIsNone(m.component('logic_to_linear'))
        self.assertFalse(m.p.active)

    def test_deactivate_empty_logical_constraint_container(self):
        m = ConcreteModel()
        m.propositions = LogicalConstraintList()
        TransformationFactory('core.logical_to_linear').apply_to(m)
        self.assertIsNone(m.component('logic_to_linear'))
        self.assertFalse(m.propositions.active)