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
class MiscVarTests(unittest.TestCase):

    def test_error1(self):
        a = Var(name='a')
        try:
            a = Var(foo=1)
            self.fail('test_error1')
        except ValueError:
            pass

    def test_getattr1(self):
        """
        Verify the behavior of non-standard suffixes with simple variable
        """
        model = AbstractModel()
        model.a = Var()
        model.suffix = Suffix(datatype=Suffix.INT)
        instance = model.create_instance()
        self.assertEqual(instance.suffix.get(instance.a), None)
        instance.suffix.set_value(instance.a, True)
        self.assertEqual(instance.suffix.get(instance.a), True)

    def test_getattr2(self):
        """
        Verify the behavior of non-standard suffixes with an array of variables
        """
        model = AbstractModel()
        model.X = Set(initialize=[1, 3, 5])
        model.a = Var(model.X)
        model.suffix = Suffix(datatype=Suffix.INT)
        try:
            self.assertEqual(model.a.suffix, None)
            self.fail('Expected AttributeError')
        except AttributeError:
            pass
        instance = model.create_instance()
        self.assertEqual(instance.suffix.get(instance.a[1]), None)
        instance.suffix.set_value(instance.a[1], True)
        self.assertEqual(instance.suffix.get(instance.a[1]), True)

    def test_error2(self):
        try:
            model = AbstractModel()
            model.a = Var(initialize=[1, 2, 3])
            model.b = Var(model.a)
            self.fail('test_error2')
        except TypeError:
            pass

    def test_contains(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, dense=True)
        instance = model.create_instance()
        self.assertEqual(1 in instance.b, True)

    def test_float_int(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, initialize=1.1)
        model.c = Var(initialize=2.1)
        model.d = Var()
        instance = model.create_instance()
        self.assertEqual(float(value(instance.b[1])), 1.1)
        self.assertEqual(int(value(instance.b[1])), 1)
        self.assertEqual(float(value(instance.c)), 2.1)
        self.assertEqual(int(value(instance.c)), 2)
        try:
            float(instance.d)
            self.fail('expected TypeError')
        except TypeError:
            pass
        try:
            int(instance.d)
            self.fail('expected TypeError')
        except TypeError:
            pass
        try:
            float(instance.b)
            self.fail('expected TypeError')
        except TypeError:
            pass
        try:
            int(instance.b)
            self.fail('expected TypeError')
        except TypeError:
            pass

    def test_set_get(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, initialize=1.1, within=PositiveReals)
        model.c = Var(initialize=2.1, within=PositiveReals, bounds=(1, 10))
        with self.assertRaisesRegex(ValueError, "Cannot set the value for the indexed component 'b' without specifying an index value"):
            model.b = 2.2
        instance = model.create_instance()
        with self.assertRaisesRegex(KeyError, "Cannot treat the scalar component 'c' as an indexed component"):
            instance.c[1] = 2.2
        instance.b[1] = 2.2
        with self.assertRaisesRegex(KeyError, "Index '4' is not valid for indexed component 'b'"):
            instance.b[4] = 2.2
        with LoggingIntercept() as LOG:
            instance.b[3] = -2.2
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'b[3]' to a value `-2.2` (float) not in domain PositiveReals.")
        with self.assertRaisesRegex(KeyError, "Cannot treat the scalar component 'c' as an indexed component"):
            tmp = instance.c[3]
        with LoggingIntercept() as LOG:
            instance.c = 'a'
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'c' to a value `a` (str) not in domain PositiveReals.")
        with LoggingIntercept() as LOG:
            instance.c = -2.2
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'c' to a value `-2.2` (float) not in domain PositiveReals.")
        with LoggingIntercept() as LOG:
            instance.c = 11
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'c' to a numeric value `11` outside the bounds (1, 10).")
        with LoggingIntercept() as LOG:
            instance.c.set_value('a', skip_validation=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(instance.c.value, 'a')
        with LoggingIntercept() as LOG:
            instance.c.set_value(-1, skip_validation=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(instance.c.value, -1)

    def test_set_index(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1, 2, 3])
        model.x = Var(model.s, initialize=0, dense=True)
        self.assertEqual(len(model.x), 3)
        for i in model.s:
            self.assertEqual(value(model.x[i]), 0)
        model.s.add(4)
        self.assertEqual(len(model.x), 3)
        for i in model.s:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)

    def test_simple_default_domain(self):
        model = ConcreteModel()
        model.x = Var()
        self.assertIs(model.x.domain, Reals)

    def test_simple_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = Var(domain=Integers)
        self.assertIs(model.x.domain, Integers)

    def test_simple_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.x = Var(domain=25)

    def test_simple_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = Var(domain=lambda m: Integers)
        self.assertIs(model.x.domain, Integers)

    def test_simple_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.x = Var(domain=lambda m: 25)

    def test_indexed_default_domain(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s)
        self.assertIs(model.x[1].domain, Reals)

    def test_indexed_nondefault_domain_value(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s, domain=Integers)
        self.assertIs(model.x[1].domain, Integers)

    def test_indexed_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        with self.assertRaises(TypeError):
            model.x = Var(model.s, domain=25)

    def test_indexed_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.x = Var(model.s, domain=lambda m, i: Integers)
        self.assertIs(model.x[1].domain, Integers)

    def test_indexed_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        with self.assertRaises(TypeError):
            model.x = Var(model.s, domain=lambda m, i: 25)

    def test_list_default_domain(self):
        model = ConcreteModel()
        model.x = VarList()
        model.x.add()
        self.assertIs(model.x[1].domain, Reals)

    def test_list_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = VarList(domain=Integers)
        model.x.add()
        self.assertIs(model.x[1].domain, Integers)

    def test_list_bad_nondefault_domain_value(self):
        model = ConcreteModel()
        model.x = VarList(domain=25)
        with self.assertRaises(TypeError):
            model.x.add()

    def test_list_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: Integers)
        model.x.add()
        self.assertIs(model.x[1].domain, Integers)

    def test_list_bad_nondefault_domain_rule(self):
        model = ConcreteModel()
        model.x = VarList(domain=lambda m, i: 25)
        with self.assertRaises(TypeError):
            model.x.add()

    def test_setdata_index(self):
        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex, initialize=[1, 2, 3])
        model.x = Var(model.s[1], initialize=0, dense=True)
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        newIdx = 4
        self.assertFalse(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)
        model.s[1].add(newIdx)
        self.assertTrue(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)
        self.assertTrue(newIdx in model.s[1])
        self.assertTrue(newIdx in model.x)

    def test_setdata_multidimen_index(self):
        model = ConcreteModel()
        model.sindex = Set(initialize=[1])
        model.s = Set(model.sindex, dimen=2, initialize=[(1, 1), (1, 2), (1, 3)])
        model.x = Var(model.s[1], initialize=0, dense=True)
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        newIdx = (1, 4)
        self.assertFalse(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)
        model.s[1].add(newIdx)
        self.assertTrue(newIdx in model.s[1])
        self.assertFalse(newIdx in model.x)
        self.assertEqual(len(model.x), 3)
        for i in model.s[1]:
            self.assertEqual(value(model.x[i]), 0)
        self.assertEqual(len(model.x), 4)
        self.assertTrue(newIdx in model.s[1])
        self.assertTrue(newIdx in model.x)

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Var(model.C)

    @unittest.skipUnless(pint_available, 'units test requires pint module')
    def test_set_value_units(self):
        m = ConcreteModel()
        m.x = Var(units=units.g)
        m.x = 5
        self.assertEqual(value(m.x), 5)
        m.x = 6 * units.g
        self.assertEqual(value(m.x), 6)
        m.x = None
        self.assertIsNone(m.x.value, None)
        m.x = 7 * units.kg
        self.assertEqual(value(m.x), 7000)
        with self.assertRaises(UnitsError):
            m.x = 1 * units.s
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), '\n1 Var Declarations\n    x : Size=1, Index=None, Units=g\n        Key  : Lower : Value  : Upper : Fixed : Stale : Domain\n        None :  None : 7000.0 :  None : False : False :  Reals\n\n1 Declarations: x\n        '.strip())

    @unittest.skipUnless(pint_available, 'units test requires pint module')
    def test_set_bounds_units(self):
        m = ConcreteModel()
        m.x = Var(units=units.g)
        m.p = Param(mutable=True, initialize=1, units=units.kg)
        m.x.setlb(5)
        self.assertEqual(m.x.lb, 5)
        m.x.setlb(6 * units.g)
        self.assertEqual(m.x.lb, 6)
        m.x.setlb(7 * units.kg)
        self.assertEqual(m.x.lb, 7000)
        with self.assertRaises(UnitsError):
            m.x.setlb(1 * units.s)
        m.x.setlb(m.p)
        self.assertEqual(m.x.lb, 1000)
        m.p = 2 * units.kg
        self.assertEqual(m.x.lb, 2000)
        m.x.setub(2)
        self.assertEqual(m.x.ub, 2)
        m.x.setub(3 * units.g)
        self.assertEqual(m.x.ub, 3)
        m.x.setub(4 * units.kg)
        self.assertEqual(m.x.ub, 4000)
        with self.assertRaises(UnitsError):
            m.x.setub(1 * units.s)
        m.x.setub(m.p)
        self.assertEqual(m.x.ub, 2000)
        m.p = 3 * units.kg
        self.assertEqual(m.x.ub, 3000)

    def test_stale(self):
        m = ConcreteModel()
        m.x = Var(initialize=0)
        self.assertFalse(m.x.stale)
        m.y = Var()
        self.assertTrue(m.y.stale)
        StaleFlagManager.mark_all_as_stale(delayed=False)
        self.assertTrue(m.x.stale)
        self.assertTrue(m.y.stale)
        m.x = 1
        self.assertFalse(m.x.stale)
        self.assertTrue(m.y.stale)
        m.y = 2
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        StaleFlagManager.mark_all_as_stale(delayed=True)
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)
        m.x = 1
        self.assertFalse(m.x.stale)
        self.assertTrue(m.y.stale)
        m.y = 2
        self.assertFalse(m.x.stale)
        self.assertFalse(m.y.stale)

    def test_stale_clone(self):
        m = ConcreteModel()
        m.x = Var(initialize=0)
        self.assertFalse(m.x.stale)
        m.y = Var()
        self.assertTrue(m.y.stale)
        m.z = Var(initialize=0)
        self.assertFalse(m.z.stale)
        i = m.clone()
        self.assertFalse(i.x.stale)
        self.assertTrue(i.y.stale)
        self.assertFalse(i.z.stale)
        StaleFlagManager.mark_all_as_stale(delayed=True)
        m.z = 5
        i = m.clone()
        self.assertTrue(i.x.stale)
        self.assertTrue(i.y.stale)
        self.assertFalse(i.z.stale)

    def test_domain_categories(self):
        """Test domain attribute"""
        x = Var()
        x.construct()
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), True)
        self.assertEqual(x.bounds, (None, None))
        x.domain = Integers
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (None, None))
        x.domain = Binary
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), True)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 1))
        x.domain = RangeSet(0, 10, 0)
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), True)
        self.assertEqual(x.bounds, (0, 10))
        x.domain = RangeSet(0, 10, 1)
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 10))
        x.domain = RangeSet(0.5, 10, 1)
        self.assertEqual(x.is_integer(), False)
        self.assertEqual(x.is_binary(), False)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0.5, 9.5))
        x.domain = RangeSet(0, 1, 1)
        self.assertEqual(x.is_integer(), True)
        self.assertEqual(x.is_binary(), True)
        self.assertEqual(x.is_continuous(), False)
        self.assertEqual(x.bounds, (0, 1))