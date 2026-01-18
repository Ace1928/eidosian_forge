import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def _external_model(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    m = ConcreteModel()
    m.hypot = ExternalFunction(library=DLL, function='gsl_hypot')
    m.p = Param(initialize=1, mutable=True)
    m.x = Var(initialize=3, bounds=(1e-05, None))
    m.y = Var(initialize=3, bounds=(0, None))
    m.z = Var(initialize=1)
    m.o = Objective(expr=m.z ** 2 * m.hypot(m.p * m.x, m.p + m.y) ** 2)
    self.assertAlmostEqual(value(m.o), 25.0, 7)
    return m