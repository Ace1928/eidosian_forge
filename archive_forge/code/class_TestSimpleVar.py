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
class TestSimpleVar(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x.fixed, True)

    def Xtest_setlb_nondata_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.e = Expression()
        with self.assertRaises(ValueError):
            model.x.setlb(model.e)
        model.e.expr = 1.0
        with self.assertRaises(ValueError):
            model.x.setlb(model.e)
        model.y = Var()
        with self.assertRaises(ValueError):
            model.x.setlb(model.y)
        model.y.value = 1.0
        with self.assertRaises(ValueError):
            model.x.setlb(model.y)
        model.y.fix()
        with self.assertRaises(ValueError):
            model.x.setlb(model.y + 1)

    def Xtest_setub_nondata_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.e = Expression()
        with self.assertRaises(ValueError):
            model.x.setub(model.e)
        model.e.expr = 1.0
        with self.assertRaises(ValueError):
            model.x.setub(model.e)
        model.y = Var()
        with self.assertRaises(ValueError):
            model.x.setub(model.y)
        model.y.value = 1.0
        with self.assertRaises(ValueError):
            model.x.setub(model.y)
        model.y.fix()
        with self.assertRaises(ValueError):
            model.x.setub(model.y + 1)

    def Xtest_setlb_data_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.x.setlb(model.p)
        model.x.setlb(model.p ** 2 + 1)
        model.p.value = 1.0
        model.x.setlb(model.p)
        model.x.setlb(model.p ** 2)
        model.x.setlb(1.0)

    def Xtest_setub_data_expression(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.x.setub(model.p)
        model.x.setub(model.p ** 2 + 1)
        model.p.value = 1.0
        model.x.setub(model.p)
        model.x.setub(model.p ** 2)
        model.x.setub(1.0)

    def test_setlb_indexed(self):
        """Test setlb variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, None)
        self.instance.y.setlb(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, 1)
        self.instance.y.setlb(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].lb, None)

    def test_setub_indexed(self):
        """Test setub variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, None)
        self.instance.y.setub(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, 1)
        self.instance.y.setub(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].ub, None)

    def test_fix_all(self):
        """Test fix all variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B, dense=True)
        self.instance = self.model.create_instance()
        self.instance.fix_all_vars()
        self.assertEqual(self.instance.x.fixed, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].fixed, True)

    def test_unfix_all(self):
        """Test unfix all variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        for a in self.instance.B:
            self.instance.y[a].fixed = True
        self.instance.unfix_all_vars()
        self.assertEqual(self.instance.x.fixed, False)
        for a in self.instance.B:
            self.assertEqual(self.instance.y[a].fixed, False)

    def test_fix_indexed(self):
        """Test fix variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.y) > 0, True)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.free()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix(1)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, 1)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.unfix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, 1)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y.fix(None)
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, True)
        self.instance.y.unfix()
        for a in self.instance.y:
            self.assertEqual(self.instance.y[a].value, None)
            self.assertEqual(self.instance.y[a].fixed, False)
        self.instance.y[1].fix()
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, True)
        self.instance.y[1].free()
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, False)
        self.instance.y[1].fix(value=1)
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, True)
        self.instance.y[1].unfix()
        self.assertEqual(self.instance.y[1].value, 1)
        self.assertEqual(self.instance.y[1].fixed, False)
        self.instance.y[1].fix(value=None)
        self.assertEqual(self.instance.y[1].value, None)
        self.assertEqual(self.instance.y[1].fixed, True)

    def test_unfix_indexed(self):
        """Test unfix variables method"""
        self.model.B = RangeSet(4)
        self.model.y = Var(self.model.B)
        self.instance = self.model.create_instance()
        for a in self.instance.B:
            self.instance.y[a].fixed = True
        self.instance.unfix_all_vars()
        for a in self.instance.B:
            self.assertEqual(self.instance.y[a].fixed, False)

    def test_fix_nonindexed(self):
        """Test fix variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, True)
        self.instance.x.free()
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix(1)
        self.assertEqual(self.instance.x.value, 1)
        self.assertEqual(self.instance.x.fixed, True)
        self.instance.x.unfix()
        self.assertEqual(self.instance.x.value, 1)
        self.assertEqual(self.instance.x.fixed, False)
        self.instance.x.fix(None)
        self.assertEqual(self.instance.x.value, None)
        self.assertEqual(self.instance.x.fixed, True)

    def test_unfix_nonindexed(self):
        """Test unfix variables method"""
        self.model.B = RangeSet(4)
        self.model.x = Var()
        self.model.y = Var(self.model.B)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.instance.x.unfix()
        self.assertEqual(self.instance.x.fixed, False)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.value = 3.5
        self.assertEqual(self.instance.x.value, 3.5)

    def test_domain_attr(self):
        """Test domain attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.domain = Integers
        self.assertEqual(type(self.instance.x.domain), IntegerSet)
        self.assertEqual(self.instance.x.is_integer(), True)
        self.assertEqual(self.instance.x.is_binary(), False)
        self.assertEqual(self.instance.x.is_continuous(), False)

    def test_name_attr(self):
        """Test name attribute"""
        self.model.x = Var()
        self.model.x._name = 'foo'
        self.assertEqual(self.model.x.name, 'foo')

    def test_lb_attr1(self):
        """Test lb attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.setlb(-1.0)
        self.assertEqual(value(self.instance.x.lb), -1.0)

    def test_lb_attr2(self):
        """Test lb attribute"""
        self.model.x = Var(within=NonNegativeReals, bounds=(-1, 2))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), 0.0)
        self.assertEqual(value(self.instance.x.ub), 2.0)

    def test_lb_attr3(self):
        """Test lb attribute"""
        self.model.p = Param(mutable=True, initialize=1)
        self.model.x = Var(within=NonNegativeReals, bounds=(self.model.p, None))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), 1.0)
        self.instance.p = 2
        self.assertEqual(value(self.instance.x.lb), 2.0)

    def test_ub_attr1(self):
        """Test ub attribute"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.instance.x.setub(1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_ub_attr2(self):
        """Test ub attribute"""
        self.model.x = Var(within=NonPositiveReals, bounds=(-2, 1))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -2.0)
        self.assertEqual(value(self.instance.x.ub), 0.0)

    def test_within_option(self):
        """Test within option"""
        self.model.x = Var(within=Reals)
        self.construct()
        self.assertEqual(type(self.instance.x.domain), RealSet)
        self.assertEqual(self.instance.x.is_integer(), False)
        self.assertEqual(self.instance.x.is_binary(), False)
        self.assertEqual(self.instance.x.is_continuous(), True)

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model):
            return (-1.0, 1.0)
        self.model.x = Var(bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x.lb), -1.0)
        self.assertEqual(value(self.instance.x.ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model):
            return 1.3
        self.model.x = Var(initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model):
            return 1.3
        self.model.x = Var(initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(initialize={None: 1.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(initialize=1.3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, 1.3)
        self.instance.x = 1
        self.assertEqual(self.instance.x.value, 1)

    def test_without_initial_value(self):
        """Test default initial value"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.value, None)
        self.instance.x = 6
        self.assertEqual(self.instance.x.value, 6)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 0)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var()
        self.instance = self.model.create_instance()
        self.assertEqual(list(self.instance.x.keys()), [None])
        self.assertEqual(id(self.instance.x), id(self.instance.x[None]))

    def test_len(self):
        """Test len method"""
        self.model.x = Var()
        self.assertEqual(len(self.model.x), 0)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 1)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x.value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x.value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x.value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)