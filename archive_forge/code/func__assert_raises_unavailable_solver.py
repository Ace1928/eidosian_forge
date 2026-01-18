import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
def _assert_raises_unavailable_solver(self, solver_name):
    with self.assertRaises(RuntimeError, msg=f'The Pyomo persistent interface to {solver_name} could not be found.'):
        _test_iis(solver_name)