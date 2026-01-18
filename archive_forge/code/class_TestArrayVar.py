import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
class TestArrayVar(TestSimpleVar):

    def setUp(self):
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1, 2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A)
        self.model.y = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1].fixed, False)
        self.instance.y[1].fixed = True
        self.assertEqual(self.instance.y[1].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var(self.model.A, dense=True)
        self.model.y = Var(self.model.A, dense=True)
        self.instance = self.model.create_instance()
        try:
            self.instance.x = 3.5
            self.fail('Expected ValueError')
        except ValueError:
            pass
        self.instance.y[1] = 3.5
        self.assertEqual(self.instance.y[1].value, 3.5)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model, key):
            i = key + 11
            return key == 1 and 1.3 or 2.3
        self.model.x = Var(self.model.A, initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(self.model.A, initialize={1: 1.3, 2: 2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, 2.3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_subdict(self):
        """Test initialize option method with a dictionary of subkeys"""
        self.model.x = Var(self.model.A, initialize={1: 1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A, initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 3)
        self.assertEqual(self.instance.x[2].value, 3)
        self.instance.x[1] = 1
        self.instance.x[2] = 2
        self.assertEqual(self.instance.x[1].value, 1)
        self.assertEqual(self.instance.x[2].value, 2)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, None)
        self.assertEqual(self.instance.x[2].value, None)
        self.instance.x[1] = 5
        self.instance.x[2] = 6
        self.assertEqual(self.instance.x[1].value, 5)
        self.assertEqual(self.instance.x[2].value, 6)

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model, i):
            return (-1.0, 1.0)
        self.model.x = Var(self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model, i):
            return 1.3
        self.model.x = Var(self.model.A, initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1].value, 1.3)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(set(self.instance.x.keys()), set())
        self.assertEqual(set(self.instance.y.keys()), set([1, 2]))
        self.assertEqual(set(self.instance.z.keys()), set([1, 2]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A, dense=False)
        self.model.y = Var(self.model.A, dense=True)
        self.model.z = Var(self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 0)
        self.assertEqual(len(self.instance.y), 2)
        self.assertEqual(len(self.instance.z), 2)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A, initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x[1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x[1].value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)

    def test_var_domain_setter(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, Reals)
        self.assertIs(m.x[3].domain, Reals)
        m.x.domain = Integers
        self.assertIs(m.x[1].domain, Integers)
        self.assertIs(m.x[2].domain, Integers)
        self.assertIs(m.x[3].domain, Integers)
        m.x.domain = lambda m, i: PositiveReals
        self.assertIs(m.x[1].domain, PositiveReals)
        self.assertIs(m.x[2].domain, PositiveReals)
        self.assertIs(m.x[3].domain, PositiveReals)
        m.x.domain = {1: Reals, 2: NonPositiveReals, 3: NonNegativeReals}
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, NonPositiveReals)
        self.assertIs(m.x[3].domain, NonNegativeReals)
        m.x.domain = {2: Integers}
        self.assertIs(m.x[1].domain, Reals)
        self.assertIs(m.x[2].domain, Integers)
        self.assertIs(m.x[3].domain, NonNegativeReals)
        with LoggingIntercept() as LOG, self.assertRaisesRegex(TypeError, "Cannot create a Set from data that does not support __contains__.  Expected set-like object supporting collections.abc.Collection interface, but received 'NoneType'"):
            m.x.domain = {1: None, 2: None, 3: None}
        self.assertIn('{1: None, 2: None, 3: None} is not a valid domain. Variable domains must be an instance of a Pyomo Set or convertible to a Pyomo Set.', LOG.getvalue())