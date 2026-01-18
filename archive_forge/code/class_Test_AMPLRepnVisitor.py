import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
class Test_AMPLRepnVisitor(unittest.TestCase):

    def test_divide(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=1)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** 2 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x)]))
        m.p = 2
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 0.5})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 * m.x / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 * (m.x + 2) / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 4)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** 2 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o2\nn0.5\no5\n%s\nn2\n', [id(m.x)]))
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x) / m.x, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o3\no43\n%s\n%s\n', [id(m.x), id(m.x)]))

    def test_errors_divide_by_0(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((1 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((3 * m.x / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3*(x + 2)/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** 2 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: x**2/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x)]))
        m.p = 1
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)
        m.p = 0
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_divide_by_0_mult_by_0(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (1 / m.p), None, None, 1))
        self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((1 / m.p * m.p, None, None, 1))
        self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (m.x / m.p), None, None, 1))
        self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (3 * (m.x + 2) / m.p), None, None, 1))
        self.assertIn("Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3*(x + 2)/p\n", LOG.getvalue())
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (m.x ** 2 / m.p), None, None, 1))
        self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: x**2/p\n", LOG.getvalue())
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_divide_by_0_halt(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()
        repn_util.HALT_ON_EVALUATION_ERROR, tmp = (True, repn_util.HALT_ON_EVALUATION_ERROR)
        try:
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((1 / m.p, None, None, 1))
            self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n")
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x / m.p, None, None, 1))
            self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n")
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None, 1))
            self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3*(x + 2)/p\n")
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x ** 2 / m.p, None, None, 1))
            self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: x**2/p\n")
        finally:
            repn_util.HALT_ON_EVALUATION_ERROR = tmp

    def test_errors_negative_frac_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=-1)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p ** 0.5, None, None, 1))
        self.assertEqual(LOG.getvalue(), 'Complex number returned from expression\n\tmessage: Pyomo AMPLRepnVisitor does not support complex numbers\n\texpression: p**0.5\n')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertRegex(str(repn.const), _invalid_1j)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.x.fix(0.5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p ** m.x, None, None, 1))
        self.assertEqual(LOG.getvalue(), 'Complex number returned from expression\n\tmessage: Pyomo AMPLRepnVisitor does not support complex numbers\n\texpression: p**x\n')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertRegex(str(repn.const), _invalid_1j)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_unary_func(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.p), None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'log(0)'\n\tmessage: math domain error\n\texpression: log(p)\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_propagate_nan(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0, domain=Any)
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.y.fix(1)
        expr = m.y ** 2 * m.x ** 2 * (3 * m.x / m.p * m.x) / m.y
        with LoggingIntercept() as LOG, INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3/p\n")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.y.fix(None)
        expr = log(m.y) + 3
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        expr = 3 * m.y
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.p.value = None
        expr = 5 * (m.p * m.x + 2 * m.z)
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.z): 10, id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)
        expr = m.y * m.x
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)
        m.z = Var([1, 2, 3, 4], initialize=lambda m, i: i - 1)
        m.z[1].fix(None)
        expr = m.z[1] - m.z[2] * m.z[3] * m.z[4]
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear[0], 'o16\no2\no2\n%s\n%s\n%s\n')
        self.assertEqual(repn.nonlinear[1], [id(m.z[2]), id(m.z[3]), id(m.z[4])])
        m.z[3].fix(float('nan'))
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_linearexpression_npv(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        m.z = Var(initialize=4)
        m.p = Param(initialize=5, mutable=True)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((LinearExpression(args=[1, m.p, m.p * m.x, (m.p + 2) * m.y, 3 * m.z, m.p * m.z]), None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 6)
        self.assertEqual(repn.linear, {id(m.x): 5, id(m.y): 7, id(m.z): 8})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_pow(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** 0.5, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn0.5\n', [id(m.x)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** 0.5, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_abs(self):
        m = ConcreteModel()
        m.x = Var(initialize=-4)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((abs(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o15\n%s\n', [id(m.x)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((abs(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 4)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_unary_func(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o43\n%s\n', [id(m.x)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, math.log(4))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_lessEq(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(m.x <= 4, m.x ** 2, m.y)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o35\no23\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.y)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_Eq(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(m.x == 4, m.x ** 2, m.y)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o35\no24\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.y)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_ranged(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(inequality(1, m.x, 4), m.x ** 2, m.y)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o35\no21\no23\nn1\n%s\no23\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.x), id(m.y)]))
        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)
        m.x.fix(0)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_custom_named_expression(self):

        class CustomExpression(ScalarExpression):
            pass
        m = ConcreteModel()
        m.x = Var()
        m.e = CustomExpression()
        m.e.expr = m.x + 3
        expr = m.e + m.e
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 6)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)
        self.assertEqual(len(info.subexpression_cache), 1)
        obj, repn, info = info.subexpression_cache[id(m.e)]
        self.assertIs(obj, m.e)
        self.assertEqual(repn.nl, ('%s\n', (id(m.e),)))
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 3)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)
        self.assertEqual(info, [None, None, False])

    def test_nested_operator_zero_arg(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(initialize=0, mutable=True)
        expr = 1 / m.x == m.p
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o24\no3\nn1\n%s\nn0\n', [id(m.x)]))

    def test_duplicate_shared_linear_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=2 * m.x + 3 * m.y)
        expr1 = 10 * m.e
        expr2 = m.e + 100 * m.x + 100 * m.y
        info = INFO()
        with LoggingIntercept() as LOG:
            repn1 = info.visitor.walk_expression((expr1, None, None, 1))
            repn2 = info.visitor.walk_expression((expr2, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn1.nl, None)
        self.assertEqual(repn1.mult, 1)
        self.assertEqual(repn1.const, 0)
        self.assertEqual(repn1.linear, {id(m.x): 20, id(m.y): 30})
        self.assertEqual(repn1.nonlinear, None)
        self.assertEqual(repn2.nl, None)
        self.assertEqual(repn2.mult, 1)
        self.assertEqual(repn2.const, 0)
        self.assertEqual(repn2.linear, {id(m.x): 102, id(m.y): 103})
        self.assertEqual(repn2.nonlinear, None)

    def test_AMPLRepn_to_expr(self):
        m = ConcreteModel()
        m.p = Param([2, 3, 4], mutable=True, initialize=lambda m, i: i ** 2)
        m.x = Var([2, 3, 4], initialize=lambda m, i: i)
        e = 10
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((e, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 10)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)
        ee = repn.to_expr(info.var_map)
        self.assertExpressionsEqual(ee, 10)
        e += sum((m.x[i] * m.p[i] for i in m.x))
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((e, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 10)
        self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
        self.assertEqual(repn.nonlinear, None)
        ee = repn.to_expr(info.var_map)
        self.assertExpressionsEqual(ee, 4 * m.x[2] + 9 * m.x[3] + 16 * m.x[4] + 10)
        self.assertEqual(ee(), 10 + 8 + 27 + 64)
        e = sum((m.x[i] * m.p[i] for i in m.x))
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((e, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
        self.assertEqual(repn.nonlinear, None)
        ee = repn.to_expr(info.var_map)
        self.assertExpressionsEqual(ee, 4 * m.x[2] + 9 * m.x[3] + 16 * m.x[4])
        self.assertEqual(ee(), 8 + 27 + 64)
        e += m.x[2] ** 2
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((e, None, None, 1))
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x[2])]))
        with self.assertRaisesRegex(MouseTrap, 'Cannot convert nonlinear AMPLRepn to Pyomo Expression'):
            ee = repn.to_expr(info.var_map)