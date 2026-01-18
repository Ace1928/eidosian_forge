import pickle
import math
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable, variable_tuple
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.conic import (
class _conic_tester_base(object):
    _object_factory = None

    def setUp(self):
        assert self._object_factory is not None

    def test_pprint(self):
        c = self._object_factory()
        pprint(c)
        b = block()
        b.c = c
        pprint(c)
        pprint(b)
        m = block()
        m.b = b
        pprint(c)
        pprint(b)
        pprint(m)

    def test_type(self):
        c = self._object_factory()
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IConstraint))

    def test_ctype(self):
        c = self._object_factory()
        self.assertIs(c.ctype, IConstraint)
        self.assertIs(type(c)._ctype, IConstraint)

    def test_pickle(self):
        c = self._object_factory()
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, 0)
        self.assertIsNot(c.body, None)
        self.assertIs(c.parent, None)
        cup = pickle.loads(pickle.dumps(c))
        self.assertIs(cup.lb, None)
        self.assertEqual(cup.ub, 0)
        self.assertIsNot(cup.body, None)
        self.assertIs(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(pickle.dumps(b))
        cup = bup.c
        self.assertIs(cup.lb, None)
        self.assertEqual(cup.ub, 0)
        self.assertIsNot(cup.body, None)
        self.assertIs(cup.parent, bup)

    def test_properties(self):
        c = self._object_factory()
        self.assertIs(c._body, None)
        self.assertIs(c.parent, None)
        self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.check_convexity_conditions(), True)
        self.assertEqual(c.check_convexity_conditions(relax=False), True)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        with self.assertRaises(AttributeError):
            c.lb = 1
        with self.assertRaises(AttributeError):
            c.ub = 1
        with self.assertRaises(AttributeError):
            c.rhs = 1
        with self.assertRaises(ValueError):
            c.rhs
        with self.assertRaises(AttributeError):
            c.equality = True
        self.assertIs(c._body, None)
        self.assertIs(c.parent, None)
        self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.check_convexity_conditions(), True)
        self.assertEqual(c.check_convexity_conditions(relax=False), True)
        self.assertEqual(c.check_convexity_conditions(relax=True), True)
        self.assertEqual(c.active, True)
        with self.assertRaises(AttributeError):
            c.active = False
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        c.activate()
        self.assertEqual(c.active, True)

    def test_containers(self):
        c = self._object_factory()
        self.assertIs(c.parent, None)
        cdict = constraint_dict()
        cdict[None] = c
        self.assertIs(c.parent, cdict)
        del cdict[None]
        self.assertIs(c.parent, None)
        clist = constraint_list()
        clist.append(c)
        self.assertIs(c.parent, clist)
        clist.remove(c)
        self.assertIs(c.parent, None)
        ctuple = constraint_tuple((c,))
        self.assertIs(c.parent, ctuple)