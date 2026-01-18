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
class Test_NLWriter(unittest.TestCase):

    def test_external_function_str_args(self):
        m = ConcreteModel()
        m.x = Var()
        m.e = ExternalFunction(library='tmp', function='test')
        m.o = Objective(expr=m.e(m.x, 'str'))
        OUT = io.StringIO(newline='\r\n')
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertIn("Writing NL file containing string arguments to a text output stream with line endings other than '\\n' ", LOG.getvalue())
        with TempfileManager:
            fname = TempfileManager.create_tempfile()
            with open(fname, 'w') as OUT:
                with LoggingIntercept() as LOG:
                    nl_writer.NLWriter().write(m, OUT)
        if os.linesep == '\n':
            self.assertEqual(LOG.getvalue(), '')
        else:
            self.assertIn("Writing NL file containing string arguments to a text output stream with line endings other than '\\n' ", LOG.getvalue())
        r, w = os.pipe()
        try:
            OUT = os.fdopen(w, 'w')
            with LoggingIntercept() as LOG:
                nl_writer.NLWriter().write(m, OUT)
            if os.linesep == '\n':
                self.assertEqual(LOG.getvalue(), '')
            else:
                self.assertIn('Writing NL file containing string arguments to a text output stream that does not support tell()', LOG.getvalue())
        finally:
            OUT.close()
            os.close(r)

    def test_suffix_warning_new_components(self):
        m = ConcreteModel()
        m.junk = Suffix(direction=Suffix.EXPORT)
        m.x = Var()
        m.y = Var()
        m.z = Var([1, 2, 3])
        m.o = Objective(expr=m.x + m.z[2])
        m.c = Constraint(expr=m.y <= 0)
        m.c.deactivate()

        @m.Constraint([1, 2, 3])
        def d(m, i):
            return m.z[i] <= 0
        m.d.deactivate()
        m.d[2].activate()
        m.junk[m.x] = 1
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(LOG.getvalue(), '')
        m.junk[m.y] = 1
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 1 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 1 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\ty\n", LOG.getvalue())
        m.junk[m.z] = 1
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 3 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 3 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\ty\n\tz[1]\n\tz[3]\n", LOG.getvalue())
        m.junk[m.c] = 2
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 4 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
        m.junk[m.d] = 2
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
        m.junk[5] = 5
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\nmodel contains export suffix 'junk' that contains 1 keys that are not Var, Constraint, Objective, or the model.  Skipping.\n", LOG.getvalue())
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\tc\n\td[1]\n\td[3]\n\ty\n\tz[1]\n\tz[3]\nmodel contains export suffix 'junk' that contains 1 keys that are not Var, Constraint, Objective, or the model.  Skipping.\nSkipped component keys:\n\t5\n", LOG.getvalue())

    def test_log_timing(self):
        m = ConcreteModel()
        m.x = Var(range(6))
        m.x[0].domain = pyo.Binary
        m.x[1].domain = pyo.Integers
        m.x[2].domain = pyo.Integers
        m.p = Param(initialize=5, mutable=True)
        m.o1 = Objective([1, 2], rule=lambda m, i: 1)
        m.o2 = Objective(expr=m.x[1] * m.x[2])
        m.c1 = Constraint([1, 2], rule=lambda m, i: sum(m.x.values()) == 1)
        m.c2 = Constraint(expr=m.p * m.x[1] ** 2 + m.x[2] ** 3 <= 100)
        self.maxDiff = None
        OUT = io.StringIO()
        with capture_output() as LOG:
            with report_timing(level=logging.DEBUG):
                nl_writer.NLWriter().write(m, OUT)
        self.assertEqual('      [+   #.##] Initialized column order\n      [+   #.##] Collected suffixes\n      [+   #.##] Objective o1\n      [+   #.##] Objective o2\n      [+   #.##] Constraint c1\n      [+   #.##] Constraint c2\n      [+   #.##] Categorized model variables: 14 nnz\n      [+   #.##] Set row / column ordering: 6 var [3, 1, 2 R/B/Z], 3 con [2, 1 L/NL]\n      [+   #.##] Generated row/col labels & comments\n      [+   #.##] Wrote NL stream\n      [    #.##] Generated NL representation\n', re.sub('\\d\\.\\d\\d\\]', '#.##]', LOG.getvalue()))

    def test_linear_constraint_npv_const(self):
        m = ConcreteModel()
        m.x = Var([1, 2])
        m.p = Param(initialize=5, mutable=True)
        m.o = Objective(expr=1)
        m.c = Constraint(expr=LinearExpression([m.p ** 2, 5 * m.x[1], 10 * m.x[2]]) <= 0)
        OUT = io.StringIO()
        nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 2 1 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 2 0 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\nn0\nO0 0\nn1.0\nx0\nr\n1 -25\nb\n3\n3\nk1\n1\nJ0 2\n0 5\n1 10\n', OUT.getvalue()))

    def test_indexed_sos_constraints(self):
        m = pyo.ConcreteModel()
        m.A = pyo.Set(initialize=[1])
        m.B = pyo.Set(initialize=[1, 2, 3])
        m.C = pyo.Set(initialize=[1])
        m.param_cx = pyo.Param(m.A, initialize={1: 1})
        m.param_cy = pyo.Param(m.B, initialize={1: 2, 2: 3, 3: 1})
        m.x = pyo.Var(m.A, domain=pyo.NonNegativeReals, bounds=(0, 40))
        m.y = pyo.Var(m.B, domain=pyo.NonNegativeIntegers)

        @m.Objective()
        def OBJ(m):
            return sum((m.param_cx[a] * m.x[a] for a in m.A)) + sum((m.param_cy[b] * m.y[b] for b in m.B))
        m.y[3].bounds = (2, 3)
        m.mysos = pyo.SOSConstraint(m.C, var=m.y, sos=1, index={1: [2, 3]}, weights={2: 25.0, 3: 18.0})
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(*nl_diff("g3 1 1 0        # problem unknown\n 4 0 1 0 0      # vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0    # nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0    # network constraints: nonlinear, linear\n 0 0 0  # nonlinear vars in constraints, objectives, both\n 0 0 0 1        # linear network variables; functions; arith, flags\n 0 3 0 0 0      # discrete variables: binary, integer, nonlinear (b,c,o)\n 0 4    # nonzeros in Jacobian, obj. gradient\n 3 4    # max name lengths: constraints, variables\n 0 0 0 0 0      # common exprs: b,c,o,c1,o1\nS0 2 sosno\n2 1\n3 1\nS0 2 ref\n2 25.0\n3 18.0\nO0 0    #OBJ\nn0\nx0      # initial guess\nr       #0 ranges (rhs's)\nb       #4 bounds (on variables)\n0 0 40  #x[1]\n2 0     #y[1]\n2 0     #y[2]\n0 2 3   #y[3]\nk3      #intermediate Jacobian column lengths\n0\n0\n0\nG0 4    #OBJ\n0 1\n1 2\n2 3\n3 1\n", OUT.getvalue()))

    @unittest.skipUnless(numpy_available, 'test requires numpy')
    def test_nonfloat_constants(self):
        import pyomo.environ as pyo
        v = numpy.array([[8], [3], [6], [11]])
        w = numpy.array([[5], [7], [4], [3]])
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=range(4))
        m.zero = pyo.Param(initialize=numpy.array([0]), mutable=True)
        m.one = pyo.Param(initialize=numpy.array([1]), mutable=True)
        m.x = pyo.Var(m.I, bounds=(m.zero, m.one), domain=pyo.Integers, initialize=True)
        m.limit = pyo.Param(initialize=numpy.array([14]), mutable=True)
        m.v = pyo.Param(m.I, initialize=v, mutable=True)
        m.w = pyo.Param(m.I, initialize=w, mutable=True)
        m.value = pyo.Objective(expr=pyo.sum_product(m.v, m.x), sense=pyo.maximize)
        m.weight = pyo.Constraint(expr=pyo.sum_product(m.w, m.x) <= m.limit)
        OUT = io.StringIO()
        ROW = io.StringIO()
        COL = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT, ROW, COL, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(ROW.getvalue(), 'weight\nvalue\n')
        self.assertEqual(COL.getvalue(), 'x[0]\nx[1]\nx[2]\nx[3]\n')
        self.assertEqual(*nl_diff("g3 1 1 0       #problem unknown\n 4 1 1 0 0     #vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0   #nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0   #network constraints: nonlinear, linear\n 0 0 0 #nonlinear vars in constraints, objectives, both\n 0 0 0 1       #linear network variables; functions; arith, flags\n 0 4 0 0 0     #discrete variables: binary, integer, nonlinear (b,c,o)\n 4 4   #nonzeros in Jacobian, obj. gradient\n 6 4   #max name lengths: constraints, variables\n 0 0 0 0 0     #common exprs: b,c,o,c1,o1\nC0     #weight\nn0\nO0 1   #value\nn0\nx4     #initial guess\n0 1.0  #x[0]\n1 1.0  #x[1]\n2 1.0  #x[2]\n3 1.0  #x[3]\nr      #1 ranges (rhs's)\n1 14.0 #weight\nb      #4 bounds (on variables)\n0 0 1  #x[0]\n0 0 1  #x[1]\n0 0 1  #x[2]\n0 0 1  #x[3]\nk3     #intermediate Jacobian column lengths\n1\n2\n3\nJ0 4   #weight\n0 5\n1 7\n2 4\n3 3\nG0 4   #value\n0 8\n1 3\n2 6\n3 11\n", OUT.getvalue()))

    def test_presolve_lower_triangular(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(nlinfo.eliminated_vars, [(m.x[3], -4.0), (m.x[1], 4.0), (m.x[2], 3.0), (m.x[0], 5.0)])
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 0 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nO0 0\nn-4.0\nx0\nr\nb\n0 -10 10\nk0\nG0 1\n0 1\n', OUT.getvalue()))

    def test_presolve_lower_triangular_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.x[0].bounds = (5, 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(nlinfo.eliminated_vars, [(m.x[3], -4.0), (m.x[1], 4.0), (m.x[2], 3.0), (m.x[0], 5.0)])
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 0 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nO0 0\nn-4.0\nx0\nr\nb\n0 -10 10\nk0\nG0 1\n0 1\n', OUT.getvalue()))

    def test_presolve_lower_triangular_implied(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(6), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == m.x[5])
        m.x[0].bounds = (None, 5)
        m.x[5].bounds = (5, None)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(nlinfo.eliminated_vars, [(m.x[1], 4.0), (m.x[5], 5.0), (m.x[3], -4.0), (m.x[2], 3.0), (m.x[0], 5.0)])
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 0 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nO0 0\nn-4.0\nx0\nr\nb\n0 -10 10\nk0\nG0 1\n0 1\n', OUT.getvalue()))

    def test_presolve_almost_lower_triangular(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] + 2 * m.x[4] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertIs(nlinfo.eliminated_vars[0][0], m.x[4])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[0][1], 3.0 * m.x[1] - 12.0)
        self.assertIs(nlinfo.eliminated_vars[1][0], m.x[3])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[1][1], 17.0 * m.x[1] - 72.0)
        self.assertIs(nlinfo.eliminated_vars[2][0], m.x[2])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[2][1], 4.0 * m.x[1] - 13.0)
        self.assertIs(nlinfo.eliminated_vars[3][0], m.x[0])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[3][1], -6.0 * m.x[1] + 29.0)
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 0 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nO0 0\nn-84.0\nx0\nr\nb\n0 3.6470588235294117 4.823529411764706\nk0\nG0 1\n0 20\n', OUT.getvalue()))

    def test_presolve_almost_lower_triangular_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4] + pyo.log(m.x[0]))
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] + 2 * m.x[4] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        m.c.add(2 * m.x[0] ** 2 + m.x[0] + m.x[2] + 3 * m.x[3] ** 3 == 10)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertIs(nlinfo.eliminated_vars[0][0], m.x[4])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[0][1], 3.0 * m.x[1] - 12.0)
        self.assertIs(nlinfo.eliminated_vars[1][0], m.x[3])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[1][1], 17.0 * m.x[1] - 72.0)
        self.assertIs(nlinfo.eliminated_vars[2][0], m.x[2])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[2][1], 4.0 * m.x[1] - 13.0)
        self.assertIs(nlinfo.eliminated_vars[3][0], m.x[0])
        self.assertExpressionsEqual(nlinfo.eliminated_vars[3][1], -6.0 * m.x[1] + 29.0)
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 1 1 0 1 \t# vars, constraints, objectives, ranges, eqns\n 1 1 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 1 1 1 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 1 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\no0\no2\nn2\no5\no0\no2\nn-6.0\nv0\nn29.0\nn2\no2\nn3\no5\no0\no2\nn17.0\nv0\nn-72.0\nn3\nO0 0\no0\no43\no0\no2\nn-6.0\nv0\nn29.0\nn-84.0\nx0\nr\n4 -6.0\nb\n0 3.6470588235294117 4.823529411764706\nk0\nJ0 1\n0 -2.0\nG0 1\n0 20.0\n', OUT.getvalue()))

    def test_presolve_lower_triangular_out_of_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), domain=pyo.NonNegativeReals)
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        OUT = io.StringIO()
        with self.assertRaisesRegex(nl_writer.InfeasibleConstraintException, "model contains a trivially infeasible variable 'x\\[3\\]' \\(presolved to a value of -4.0 outside bounds \\[0, None\\]\\)."):
            with LoggingIntercept() as LOG:
                nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')

    def test_presolve_named_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1, bounds=(0, 10))
        m.subexpr = pyo.Expression(pyo.Integers)
        m.subexpr[1] = m.x[1] + m.x[2]
        m.eq = pyo.Constraint(pyo.Integers)
        m.eq[1] = m.x[1] == 7
        m.eq[2] = m.x[3] == 0.1 * m.subexpr[1] * m.x[2]
        m.obj = pyo.Objective(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 3)
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertEqual(nlinfo.eliminated_vars, [(m.x[1], 7)])
        self.assertEqual(*nl_diff("g3 1 1 0\t# problem unknown\n 2 1 1 0 1 \t# vars, constraints, objectives, ranges, eqns\n 1 1 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 1 2 1 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 2 2 \t# nonzeros in Jacobian, obj. gradient\n 5 4\t# max name lengths: constraints, variables\n 0 0 0 1 0\t# common exprs: b,c,o,c1,o1\nV2 1 1\t#subexpr[1]\n0 1\nn7.0\nC0\t#eq[2]\no16\t#-\no2\t#*\no2\t#*\nn0.1\nv2\t#subexpr[1]\nv0\t#x[2]\nO0 0\t#obj\no54\t# sumlist\n3\t# (n)\no5\t#^\nn7.0\nn2\no5\t#^\nv0\t#x[2]\nn2\no5\t#^\nv1\t#x[3]\nn3\nx2\t# initial guess\n0 1\t#x[2]\n1 1\t#x[3]\nr\t#1 ranges (rhs's)\n4 0\t#eq[2]\nb\t#2 bounds (on variables)\n0 0 10\t#x[2]\n0 0 10\t#x[3]\nk1\t#intermediate Jacobian column lengths\n1\nJ0 2\t#eq[2]\n0 0\n1 1\nG0 2\t#obj\n0 0\n1 0\n", OUT.getvalue()))

    def test_scaling(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0)
        m.y = pyo.Var(initialize=0, bounds=(-200000.0, 100000.0))
        m.z = pyo.Var(initialize=0, bounds=(1000.0, None))
        m.v = pyo.Var(initialize=0, bounds=(1000.0, 1000.0))
        m.w = pyo.Var(initialize=0, bounds=(None, 1000.0))
        m.obj = pyo.Objective(expr=m.x ** 2 + (m.y - 50000) ** 2 + m.z)
        m.c = pyo.ConstraintList()
        m.c.add(100 * m.x + m.y / 100 >= 600)
        m.c.add(1000 * m.w + m.v * m.x <= 100)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        m.dual[m.c[1]] = 0.02
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, scale_model=False, linear_presolve=False)
        self.assertEqual(LOG.getvalue(), '')
        nl1 = OUT.getvalue()
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 5 2 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 1 1 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 2 3 1 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 5 3 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\no2\nv1\nv0\nC1\nn0\nO0 0\no0\no5\nv0\nn2\no5\no0\nv2\nn-50000\nn2\nd1\n1 0.02\nx5\n0 0\n1 0\n2 0\n3 0\n4 0\nr\n1 100\n2 600\nb\n3\n4 1000.0\n0 -200000.0 100000.0\n2 1000.0\n1 1000.0\nk4\n2\n3\n4\n4\nJ0 3\n0 0\n1 0\n4 1000\nJ1 2\n0 100\n2 0.01\nG0 3\n0 0\n2 0\n3 1\n', nl1))
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.v] = 1 / 250
        m.scaling_factor[m.w] = 1 / 500
        m.scaling_factor[m.y] = -1 / 50000
        m.scaling_factor[m.z] = 1 / 1000
        m.scaling_factor[m.c[1]] = 1 / 10
        m.scaling_factor[m.c[2]] = -1 / 100
        m.scaling_factor[m.obj] = 1 / 100
        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, scale_model=True, linear_presolve=False)
        self.assertEqual(LOG.getvalue(), '')
        nl2 = OUT.getvalue()
        self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 5 2 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 1 1 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 2 3 1 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 5 3 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\no2\nn-0.01\no2\no3\nv1\nn0.004\nv0\nC1\nn0\nO0 0\no2\nn0.01\no0\no5\nv0\nn2\no5\no0\no3\nv2\nn-2e-05\nn-50000\nn2\nd1\n1 0.002\nx5\n0 0\n1 0.0\n2 0.0\n3 0.0\n4 0.0\nr\n2 -1.0\n2 60.0\nb\n3\n4 4.0\n0 -2.0 4.0\n2 1.0\n1 2.0\nk4\n2\n3\n4\n4\nJ0 3\n0 0.0\n1 0.0\n4 -5000.0\nJ1 2\n0 10.0\n2 -50.0\nG0 3\n0 0.0\n2 0.0\n3 10.0\n', nl2))

    def test_named_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.E1 = Expression(expr=3 * (m.x * m.y + m.z))
        m.E2 = Expression(expr=m.z * m.y)
        m.E3 = Expression(expr=m.x * m.z + m.y)
        m.o1 = Objective(expr=m.E1 + m.E2)
        m.o2 = Objective(expr=m.E1 ** 2)
        m.c1 = Constraint(expr=m.E2 + 2 * m.E3 >= 0)
        m.c2 = Constraint(expr=pyo.inequality(0, m.E3 ** 2, 10))
        OUT = io.StringIO()
        nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(*nl_diff("g3 1 1 0\t# problem unknown\n 3 2 2 1 0 \t# vars, constraints, objectives, ranges, eqns\n 2 2 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 3 3 3 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 6 6 \t# nonzeros in Jacobian, obj. gradient\n 2 1\t# max name lengths: constraints, variables\n 1 1 1 1 1\t# common exprs: b,c,o,c1,o1\nV3 0 0\t#nl(E1)\no2\t#*\nv0\t#x\nv1\t#y\nV4 0 0\t#E2\no2\t#*\nv2\t#z\nv1\t#y\nV5 0 0\t#nl(E3)\no2\t#*\nv0\t#x\nv2\t#z\nC0\t#c1\no0\t#+\nv4\t#E2\no2\t#*\nn2\nv5\t#nl(E3)\nV6 1 2\t#E3\n1 1\nv5\t#nl(E3)\nC1\t#c2\no5\t#^\nv6\t#E3\nn2\nO0 0\t#o1\no0\t#+\no2\t#*\nn3\nv3\t#nl(E1)\nv4\t#E2\nV7 1 4\t#E1\n2 3\no2\t#*\nn3\nv3\t#nl(E1)\nO1 0\t#o2\no5\t#^\nv7\t#E1\nn2\nx0\t# initial guess\nr\t#2 ranges (rhs's)\n2 0\t#c1\n0 0 10\t#c2\nb\t#3 bounds (on variables)\n3\t#x\n3\t#y\n3\t#z\nk2\t#intermediate Jacobian column lengths\n2\n4\nJ0 3\t#c1\n0 0\n1 2\n2 0\nJ1 3\t#c2\n0 0\n1 0\n2 0\nG0 3\t#o1\n0 0\n1 0\n2 3\nG1 3\t#o2\n0 0\n1 0\n2 0\n", OUT.getvalue()))