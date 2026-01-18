from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
class TestAutoVars(unittest.TestCase):

    def test_synchronize_value(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)
        self.assertIsNone(m.iv.value)
        self.assertIsNone(m.biv.value)
        m.iv = True
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.iv = True
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.iv = False
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)
        m.iv = False
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)
        m.iv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.iv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.biv = 1
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.biv = 1
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10
        m.biv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.biv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.biv.value = 1 - eps
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)
        m.biv.value = 1 - eps
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)
        m.biv.value = eps
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)
        m.biv.value = eps
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)
        m.biv.value = 0.5
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)
        m.biv.value = 0.5
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)

    def test_fix_value(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)
        m.iv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())
        self.assertIsNone(m.iv.value)
        self.assertIsNone(m.biv.value)
        m.iv.fix(True)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.iv.fix(False)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)
        m.iv.fix(None)
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.biv.fix(1)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        with LoggingIntercept() as LOG:
            m.biv.fix(0.5)
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'biv' to a value `0.5` (float) not in domain Binary.")
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)
        with LoggingIntercept() as LOG:
            m.biv.fix(0.55, True)
        self.assertEqual(LOG.getvalue().strip(), '')
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.55)
        m.biv.fix(0)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)
        eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10
        with LoggingIntercept() as LOG:
            m.biv.fix(1 - eps)
        self.assertEqual(LOG.getvalue().strip(), "Setting Var 'biv' to a value `%s` (float) not in domain Binary." % (1 - eps))
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)
        with LoggingIntercept() as LOG:
            m.biv.fix(eps, True)
        self.assertEqual(LOG.getvalue().strip(), '')
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)
        m.iv.fix(True)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.iv.fix(False)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)

    def test_fix_unfix(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)
        m.iv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())
        m.iv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())
        m.iv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())
        m.biv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())
        m.biv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())
        m.biv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())

    def test_cast_to_binary(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)
        m.biv = 1
        deprecation_msg = "Implicit conversion of the Boolean indicator_var 'iv'"
        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.lb, 0)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.ub, 1)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.bounds, (0, 1))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            m.iv.lb = 1
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            m.iv.ub = 1
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            m.iv.bounds = (1, 1)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            m.iv.setlb(1)
            self.assertEqual(m.biv.lb, 1)
            m.biv.setlb(0)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            m.iv.setub(0)
            self.assertEqual(m.biv.ub, 0)
            m.biv.setub(1)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs(abs(m.iv).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo numeric value \\(biv\\) to bool'):
                bool(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(TypeError, 'Implicit conversion of Pyomo numeric value \\(biv\\) to float'):
                float(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(TypeError, 'Implicit conversion of Pyomo numeric value \\(biv\\) to int'):
                int(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((-m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs(+m.iv, m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.has_lb())
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.has_ub())
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.is_binary())
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertFalse(m.iv.is_continuous())
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.is_integer())
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.polynomial_degree(), 1)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv == 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv <= 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv >= 0).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv < 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv > 0).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            e = m.iv + 1
        assertExpressionsEqual(self, e, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.biv)), 1]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            e = m.iv - 1
        assertExpressionsEqual(self, e, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.biv)), -1]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv * 2).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv / 2).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv ** 2).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            e = 1 + m.iv
        assertExpressionsEqual(self, e, EXPR.LinearExpression([1, EXPR.MonomialTermExpression((1, m.biv))]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            e = 1 - m.iv
        assertExpressionsEqual(self, e, EXPR.LinearExpression([1, EXPR.MonomialTermExpression((-1, m.biv))]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2 * m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2 / m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2 ** m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a += 1
        assertExpressionsEqual(self, a, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.biv)), 1]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a -= 1
        assertExpressionsEqual(self, a, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.biv)), -1]))
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a *= 2
            self.assertIs(a.args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a /= 2
            self.assertIs(a.args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())
        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a **= 2
            self.assertIs(a.args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())