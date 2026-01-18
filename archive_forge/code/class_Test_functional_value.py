import pickle
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import dill, dill_available as has_dill
from pyomo.core.expr.numvalue import (
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
class Test_functional_value(unittest.TestCase):

    def test_pprint(self):
        f = functional_value()
        pprint(f)
        b = block()
        b.f = f
        pprint(f)
        pprint(b)
        m = block()
        m.b = b
        pprint(f)
        pprint(b)
        pprint(m)

    def test_ctype(self):
        f = functional_value()
        self.assertIs(f.ctype, IParameter)
        self.assertIs(type(f), functional_value)
        self.assertIs(type(f)._ctype, IParameter)

    def test_pickle(self):
        f = functional_value()
        self.assertIs(f.fn, None)
        self.assertIs(f.parent, None)
        fup = pickle.loads(pickle.dumps(f))
        self.assertIs(fup.fn, None)
        self.assertIs(fup.parent, None)
        b = block()
        b.f = f
        self.assertIs(f.parent, b)
        bup = pickle.loads(pickle.dumps(b))
        fup = bup.f
        self.assertIs(fup.fn, None)
        self.assertIs(fup.parent, bup)

    @unittest.skipIf(not has_dill, 'The dill module is not available')
    def test_dill(self):
        p = parameter(1)
        f = functional_value(lambda: p())
        self.assertEqual(f(), 1)
        fup = dill.loads(dill.dumps(f))
        p.value = 2
        self.assertEqual(f(), 2)
        self.assertEqual(fup(), 1)
        b = block()
        b.p = p
        b.f = f
        self.assertEqual(b.f(), 2)
        bup = dill.loads(dill.dumps(b))
        fup = bup.f
        b.p.value = 4
        self.assertEqual(b.f(), 4)
        self.assertEqual(bup.f(), 2)
        bup.p.value = 4
        self.assertEqual(bup.f(), 4)

    def test_call(self):
        f = functional_value()
        self.assertEqual(f(), None)
        self.assertIs(f.fn, None)
        f.fn = lambda: variable(value=1)
        self.assertIsNot(f.fn, None)
        with self.assertRaises(TypeError):
            f(exception=False)
        with self.assertRaises(TypeError):
            f(exception=True)
        with self.assertRaises(TypeError):
            f()
        f.fn = lambda: None
        self.assertIsNot(f.fn, None)
        with self.assertRaises(TypeError):
            f(exception=False)
        with self.assertRaises(TypeError):
            f(exception=True)
        with self.assertRaises(TypeError):
            f()

        def value_error():
            raise ValueError()
        f.fn = value_error
        self.assertIsNot(f.fn, None)
        self.assertEqual(f(exception=False), None)
        with self.assertRaises(ValueError):
            f(exception=True)
        with self.assertRaises(ValueError):
            f()

    def test_init(self):
        f = functional_value()
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IParameter)
        self.assertEqual(f.fn, None)
        self.assertEqual(f(), None)
        x = [1, 2]
        f.fn = lambda: max(x)
        self.assertEqual(f(), 2)
        x[0] = 3
        self.assertEqual(f(), 3)

    def test_type(self):
        f = functional_value()
        self.assertTrue(isinstance(f, ICategorizedObject))
        self.assertTrue(isinstance(f, IParameter))
        self.assertTrue(isinstance(f, NumericValue))

    def test_is_constant(self):
        f = functional_value()
        self.assertEqual(f.is_constant(), False)
        self.assertEqual(is_constant(f), False)
        f.fn = lambda: 2
        self.assertEqual(f.is_constant(), False)
        self.assertEqual(is_constant(f), False)

    def test_is_fixed(self):
        f = functional_value()
        self.assertEqual(f.is_fixed(), True)
        self.assertEqual(is_fixed(f), True)
        f.fn = lambda: 2
        self.assertEqual(f.is_fixed(), True)
        self.assertEqual(is_fixed(f), True)

    def test_potentially_variable(self):
        f = functional_value()
        self.assertEqual(f.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(f), False)
        f.fn = lambda: 2
        self.assertEqual(f.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(f), False)

    def test_polynomial_degree(self):
        f = functional_value()
        self.assertEqual(f.polynomial_degree(), 0)
        self.assertEqual((f ** 2).polynomial_degree(), 0)
        self.assertIs(f.fn, None)
        with self.assertRaises(ValueError):
            (f ** 2)()
        f.fn = lambda: 2
        self.assertEqual(f.polynomial_degree(), 0)
        self.assertEqual((f ** 2).polynomial_degree(), 0)
        self.assertEqual(f(), 2)
        self.assertEqual((f ** 2)(), 4)

    def test_is_expression_type(self):
        f = functional_value()
        self.assertEqual(f.is_expression_type(), False)

    def test_is_parameter_type(self):
        f = functional_value()
        self.assertEqual(f.is_parameter_type(), False)