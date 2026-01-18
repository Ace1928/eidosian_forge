import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
@unittest.skipIf(not mcpp_available(), 'MC++ is not available')
class TestMcCormick(unittest.TestCase):

    def test_outofbounds(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 5), initialize=2)
        with self.assertRaisesRegex(MCPP_Error, '.*Log with negative values in range'):
            mc(log(m.x))

    def test_mc_2d(self):
        m = ConcreteModel()
        m.x = Var(bounds=(pi / 6, pi / 3), initialize=pi / 4)
        m.e = Expression(expr=cos(pow(m.x, 2)) * sin(pow(m.x, -3)))
        mc_ccVals, mc_cvVals, aff_cc, aff_cv = make2dPlot(m.e.expr, 50)
        self.assertAlmostEqual(mc_ccVals[1], 0.6443888590411435)
        self.assertAlmostEqual(mc_cvVals[1], 0.2328315489072924)
        self.assertAlmostEqual(aff_cc[1], 0.9674274332870583)
        self.assertAlmostEqual(aff_cv[1], -1.578938503009686)

    def test_mc_3d(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-2, 1), initialize=-1)
        m.y = Var(bounds=(-1, 2), initialize=0)
        m.e = Expression(expr=m.x * pow(exp(m.x) - m.y, 2))
        ccSurf, cvSurf, ccAffine, cvAffine = make3dPlot(m.e.expr, 30)
        self.assertAlmostEqual(ccSurf[48], 11.5655473482574)
        self.assertAlmostEqual(cvSurf[48], -15.28102124928224)
        self.assertAlmostEqual(ccAffine[48], 11.565547348257398)
        self.assertAlmostEqual(cvAffine[48], -23.131094696514797)

    def test_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1), initialize=3)
        mc_var = mc(m.x)
        self.assertEqual(mc_var.lower(), -1)
        self.assertEqual(mc_var.upper(), 1)
        m.no_ub = Var(bounds=(0, None), initialize=3)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.mcpp', logging.WARNING):
            mc_var = mc(m.no_ub)
            self.assertIn('Var no_ub missing upper bound.', output.getvalue().strip())
            self.assertEqual(mc_var.lower(), 0)
            self.assertEqual(mc_var.upper(), 500000)
        m.no_lb = Var(bounds=(None, -3), initialize=-1)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.mcpp', logging.WARNING):
            mc_var = mc(m.no_lb)
            self.assertIn('Var no_lb missing lower bound.', output.getvalue().strip())
            self.assertEqual(mc_var.lower(), -500000)
            self.assertEqual(mc_var.upper(), -3)
        m.no_val = Var(bounds=(0, 1))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.mcpp', logging.WARNING):
            mc_var = mc(m.no_val)
            mc_var.subcv()
            self.assertIn('Var no_val missing value.', output.getvalue().strip())
            self.assertEqual(mc_var.lower(), 0)
            self.assertEqual(mc_var.upper(), 1)

    def test_fixed_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-50, 80), initialize=3)
        m.y = Var(bounds=(0, 6), initialize=2)
        m.y.fix()
        mc_expr = mc(m.x * m.y)
        self.assertEqual(mc_expr.lower(), -100)
        self.assertEqual(mc_expr.upper(), 160)
        self.assertEqual(str(mc_expr), '[ -1.00000e+02 :  1.60000e+02 ] [  6.00000e+00 :  6.00000e+00 ] [ ( 2.00000e+00) : ( 2.00000e+00) ]')

    def test_reciprocal(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 2), initialize=1)
        m.y = Var(bounds=(2, 3), initialize=2)
        mc_expr = mc(m.x / m.y)
        self.assertEqual(mc_expr.lower(), 1 / 3)
        self.assertEqual(mc_expr.upper(), 1)

    def test_nonpyomo_numeric(self):
        mc_expr = mc(-2)
        self.assertEqual(mc_expr.lower(), -2)
        self.assertEqual(mc_expr.upper(), -2)

    def test_linear_expression(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 2), initialize=1)
        mc_expr = mc(quicksum([m.x, m.x], linear=True))
        self.assertEqual(mc_expr.lower(), 2)
        self.assertEqual(mc_expr.upper(), 4)

    def test_trig(self):
        m = ConcreteModel()
        m.x = Var(bounds=(pi / 4, pi / 2), initialize=pi / 4)
        mc_expr = mc(tan(atan(m.x)))
        self.assertAlmostEqual(mc_expr.lower(), pi / 4)
        self.assertAlmostEqual(mc_expr.upper(), pi / 2)
        m.y = Var(bounds=(0, sin(pi / 4)), initialize=0)
        mc_expr = mc(asin(m.y))
        self.assertEqual(mc_expr.lower(), 0)
        self.assertAlmostEqual(mc_expr.upper(), pi / 4)
        m.z = Var(bounds=(0, cos(pi / 4)), initialize=0)
        mc_expr = mc(acos(m.z))
        self.assertAlmostEqual(mc_expr.lower(), pi / 4)
        self.assertAlmostEqual(mc_expr.upper(), pi / 2)

    def test_abs(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1), initialize=0)
        mc_expr = mc(abs(m.x))
        self.assertEqual(mc_expr.lower(), 0)
        self.assertEqual(mc_expr.upper(), 1)

    def test_lmtd(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0.1, 500), initialize=33.327)
        m.y = Var(bounds=(0.1, 500), initialize=14.436)
        m.z = Var(bounds=(0, 90), initialize=22.5653)
        e = m.z - (m.x * m.y * (m.x + m.y) / 2) ** (1 / 3)
        mc_expr = mc(e)
        for _x in [m.x.lb, m.x.ub]:
            m.x.value = _x
            mc_expr.changePoint(m.x, _x)
            for _y in [m.y.lb, m.y.ub]:
                m.y.value = _y
                mc_expr.changePoint(m.y, _y)
                for _z in [m.z.lb, m.z.ub]:
                    m.z.value = _z
                    mc_expr.changePoint(m.z, _z)
                    self.assertGreaterEqual(mc_expr.concave() + 1e-08, value(e))
                    self.assertLessEqual(mc_expr.convex() - 1e-06, value(e))
        m.x.value = m.x.lb
        m.y.value = m.y.lb
        m.z.value = m.z.lb
        mc_expr.changePoint(m.x, m.x.value)
        mc_expr.changePoint(m.y, m.y.value)
        mc_expr.changePoint(m.z, m.z.value)
        self.assertAlmostEqual(mc_expr.convex(), value(e))
        self.assertAlmostEqual(mc_expr.concave(), value(e))
        m.x.value = m.x.ub
        m.y.value = m.y.ub
        m.z.value = m.z.ub
        mc_expr.changePoint(m.x, m.x.value)
        mc_expr.changePoint(m.y, m.y.value)
        mc_expr.changePoint(m.z, m.z.value)
        self.assertAlmostEqual(mc_expr.convex(), value(e))
        self.assertAlmostEqual(mc_expr.concave(), value(e))
        self.assertAlmostEqual(mc_expr.lower(), -500)
        self.assertAlmostEqual(mc_expr.upper(), 89.9)

    def test_improved_bounds(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 100), initialize=5)
        improved_bounds = ComponentMap()
        improved_bounds[m.x] = (10, 20)
        mc_expr = mc(m.x, improved_var_bounds=improved_bounds)
        self.assertEqual(mc_expr.lower(), 10)
        self.assertEqual(mc_expr.upper(), 20)

    def test_powers(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2), initialize=1)
        m.y = Var(bounds=(0.0001, 2), initialize=1)
        m.z = Var(bounds=(-1, 1), initialize=0)
        with self.assertRaisesRegex(MCPP_Error, '(Square-root with nonpositive values in range)|(Log with negative values in range)'):
            mc(m.z ** 1.5)
        mc_expr = mc(m.y ** 1.5)
        self.assertAlmostEqual(mc_expr.lower(), 0.0001 ** 1.5)
        self.assertAlmostEqual(mc_expr.upper(), 2 ** 1.5)
        mc_expr = mc(m.y ** m.x)
        self.assertAlmostEqual(mc_expr.lower(), 0.0001 ** 2)
        self.assertAlmostEqual(mc_expr.upper(), 4)
        mc_expr = mc(m.z ** 2)
        self.assertAlmostEqual(mc_expr.lower(), 0)
        self.assertAlmostEqual(mc_expr.upper(), 1)