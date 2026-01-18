import unittest
import cvxpy as cp
from cvxpy.error import DCPError
from cvxpy.expressions.variable import Variable
class TestObjectives(unittest.TestCase):
    """ Unit tests for the expression/expression module. """

    def setUp(self) -> None:
        self.x = Variable(name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(name='z')

    def test_str(self) -> None:
        """Test string representations.
        """
        obj = cp.Minimize(self.x)
        self.assertEqual(repr(obj), 'Minimize(%s)' % repr(self.x))
        obj = cp.Minimize(2 * self.x)
        self.assertEqual(repr(obj), 'Minimize(%s)' % repr(2 * self.x))
        obj = cp.Maximize(self.x)
        self.assertEqual(repr(obj), 'Maximize(%s)' % repr(self.x))
        obj = cp.Maximize(2 * self.x)
        self.assertEqual(repr(obj), 'Maximize(%s)' % repr(2 * self.x))

    def test_minimize(self) -> None:
        exp = self.x + self.z
        obj = cp.Minimize(exp)
        self.assertEqual(str(obj), 'minimize %s' % exp.name())
        new_obj, constraints = obj.canonical_form
        self.assertEqual(len(constraints), 0)
        with self.assertRaises(Exception) as cm:
            cp.Minimize(self.y).canonical_form
        self.assertEqual(str(cm.exception), "The 'minimize' objective must resolve to a scalar.")
        copy = obj.copy()
        self.assertTrue(type(copy) is type(obj))
        self.assertEqual(copy.args, obj.args)
        self.assertFalse(copy.args is obj.args)
        copy = obj.copy(args=[cp.square(self.z)])
        self.assertTrue(type(copy) is type(obj))
        self.assertTrue(copy.args[0].args[0] is self.z)

    def test_maximize(self) -> None:
        exp = self.x + self.z
        obj = cp.Maximize(exp)
        self.assertEqual(str(obj), 'maximize %s' % exp.name())
        new_obj, constraints = obj.canonical_form
        self.assertEqual(len(constraints), 0)
        with self.assertRaises(Exception) as cm:
            cp.Maximize(self.y).canonical_form
        self.assertEqual(str(cm.exception), "The 'maximize' objective must resolve to a scalar.")
        copy = obj.copy()
        self.assertTrue(type(copy) is type(obj))
        self.assertEqual(copy.args, obj.args)
        self.assertFalse(copy.args is obj.args)
        copy = obj.copy(args=[-cp.square(self.x)])
        self.assertTrue(type(copy) is type(obj))
        self.assertTrue(copy.args[0].args[0].args[0] is self.x)

    def test_is_dcp(self) -> None:
        self.assertEqual(cp.Minimize(cp.norm_inf(self.x)).is_dcp(), True)
        self.assertEqual(cp.Minimize(-cp.norm_inf(self.x)).is_dcp(), False)
        self.assertEqual(cp.Maximize(cp.norm_inf(self.x)).is_dcp(), False)
        self.assertEqual(cp.Maximize(-cp.norm_inf(self.x)).is_dcp(), True)

    def test_add_problems(self) -> None:
        """Test adding objectives.
        """
        expr1 = self.x ** 2
        expr2 = self.x ** (-1)
        alpha = 2
        assert (cp.Minimize(expr1) + cp.Minimize(expr2)).is_dcp()
        assert (cp.Maximize(-expr1) + cp.Maximize(-expr2)).is_dcp()
        with self.assertRaises(DCPError) as cm:
            cp.Minimize(expr1) + cp.Maximize(-expr2)
        self.assertEqual(str(cm.exception), 'Problem does not follow DCP rules.')
        assert (cp.Minimize(expr1) - cp.Maximize(-expr2)).is_dcp()
        assert (alpha * cp.Minimize(expr1)).is_dcp()
        assert (alpha * cp.Maximize(-expr1)).is_dcp()
        assert (-alpha * cp.Maximize(-expr1)).is_dcp()
        assert (-alpha * cp.Maximize(-expr1)).is_dcp()