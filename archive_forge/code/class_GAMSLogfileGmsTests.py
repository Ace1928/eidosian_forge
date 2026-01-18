import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
class GAMSLogfileGmsTests(GAMSLogfileTestBase):
    """Test class for testing permultations of tee and logfile options.

    The tests build a simple model and solve it using the different options
    using the gams command directly.

    """

    def test_no_tee(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=False)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=False)

    def test_tee(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=False)

    def test_logfile(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=True)

    def test_logfile_relative(self):
        cwd = os.getcwd()
        with TempfileManager:
            tmpdir = TempfileManager.create_tempdir()
            os.chdir(tmpdir)
            try:
                self.logfile = 'test-gams.log'
                with SolverFactory('gams', solver_io='gms') as opt:
                    with capture_output() as output:
                        opt.solve(self.m, logfile=self.logfile)
                self._check_stdout(output.getvalue(), exists=False)
                self._check_logfile(exists=True)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, self.logfile)))
            finally:
                os.chdir(cwd)

    def test_tee_and_logfile(self):
        with SolverFactory('gams', solver_io='gms') as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=True)