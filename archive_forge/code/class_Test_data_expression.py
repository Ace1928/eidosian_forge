import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.objective import objective
from pyomo.core.kernel.block import block
class Test_data_expression(_Test_expression_base, unittest.TestCase):
    _ctype_factory = data_expression

    def test_associativity(self):
        x = parameter()
        y = parameter()
        pyomo.kernel.pprint(y + x * data_expression(data_expression(x * y)))
        pyomo.kernel.pprint(y + data_expression(data_expression(x * y)) * x)

    def test_ctype(self):
        e = data_expression()
        self.assertIs(e.ctype, IExpression)
        self.assertIs(type(e), data_expression)
        self.assertIs(type(e)._ctype, IExpression)

    def test_bad_init(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(e.expr, 1.0)
        v = variable()
        with self.assertRaises(ValueError):
            e = self._ctype_factory(expr=v)

    def test_bad_assignment(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(e.expr, 1.0)
        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def test_is_fixed(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        e.expr = 1
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        p = parameter()
        e.expr = p ** 2
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        a = self._ctype_factory()
        e.expr = (a * p) ** 2 / (p + 5)
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(e(), 10.0)
        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def testis_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        e.expr = 1
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        p = parameter()
        e.expr = p ** 2
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        a = self._ctype_factory()
        e.expr = (a * p) ** 2 / (p + 5)
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        self.assertEqual(e(), 10.0)
        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1

    def test_polynomial_degree(self):
        e = self._ctype_factory()
        self.assertEqual(e.polynomial_degree(), 0)
        e.expr = 1
        self.assertEqual(e.polynomial_degree(), 0)
        p = parameter()
        e.expr = p ** 2
        self.assertEqual(e.polynomial_degree(), 0)
        a = self._ctype_factory()
        e.expr = (a * p) ** 2 / (p + 5)
        self.assertEqual(e.polynomial_degree(), 0)
        a.expr = 2.0
        p.value = 5.0
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(e(), 10.0)
        v = variable()
        with self.assertRaises(ValueError):
            e.expr = v + 1