import os
from pyomo.environ import (
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
class CBCTests(unittest.TestCase):

    @unittest.skipIf(not cbc_available, 'The CBC solver is not available')
    def test_warm_start(self):
        m = ConcreteModel()
        m.x = Var()
        m.z = Var(domain=Integers)
        m.w = Var(domain=Boolean)
        m.c = Constraint(expr=m.x + m.z + m.w >= 0)
        m.o = Objective(expr=m.x + m.z + m.w)
        TempfileManager.push()
        tempdir = os.path.dirname(TempfileManager.create_tempfile())
        TempfileManager.pop()
        sameDrive = os.path.splitdrive(tempdir)[0] == os.path.splitdrive(os.getcwd())[0]
        m.x.set_value(10)
        m.z.set_value(5)
        m.w.set_value(1)
        with SolverFactory('cbc') as opt, capture_output() as output:
            opt.solve(m, tee=True, warmstart=True, options={'sloglevel': 2, 'loglevel': 2})
        log = output.getvalue()
        self.assertIn('opening mipstart file', log)
        if sameDrive:
            self.assertIn('MIPStart values read for 2 variables.', log)
            self.assertIn('MIPStart provided solution with cost 6', log)
        else:
            self.assertNotIn('MIPStart values read', log)
        m.x.set_value(10)
        m.z.set_value(5)
        m.w.set_value(1)
        try:
            _origDir = os.getcwd()
            os.chdir(tempdir)
            with SolverFactory('cbc') as opt, capture_output() as output:
                opt.solve(m, tee=True, warmstart=True, options={'sloglevel': 2, 'loglevel': 2})
        finally:
            os.chdir(_origDir)
        log = output.getvalue()
        self.assertIn('opening mipstart file', log)
        self.assertIn('MIPStart values read for 2 variables.', log)
        self.assertIn('MIPStart provided solution with cost 6', log)

    @unittest.skipIf(not cbc_available, 'The CBC solver is not available')
    def test_duals_signs(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj = Objective(expr=m.x)
        m.c = Constraint(expr=(-1, m.x, 1))
        m.dual = Suffix(direction=Suffix.IMPORT)
        opt = SolverFactory('cbc')
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, -1)
        self.assertAlmostEqual(res.problem.upper_bound, -1)
        self.assertAlmostEqual(m.dual[m.c], 1)
        m.obj.sense = maximize
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, 1)
        self.assertAlmostEqual(res.problem.upper_bound, 1)
        self.assertAlmostEqual(m.dual[m.c], 1)

    @unittest.skipIf(not cbc_available, 'The CBC solver is not available')
    def test_rc_signs(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1))
        m.obj = Objective(expr=m.x)
        m.rc = Suffix(direction=Suffix.IMPORT)
        opt = SolverFactory('cbc')
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, -1)
        self.assertAlmostEqual(res.problem.upper_bound, -1)
        self.assertAlmostEqual(m.rc[m.x], 1)
        m.obj.sense = maximize
        res = opt.solve(m)
        self.assertAlmostEqual(res.problem.lower_bound, 1)
        self.assertAlmostEqual(res.problem.upper_bound, 1)
        self.assertAlmostEqual(m.rc[m.x], 1)