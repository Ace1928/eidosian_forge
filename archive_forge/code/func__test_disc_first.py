import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
def _test_disc_first(self, tname):
    bfile = join(currdir, tname + '.' + self.sim_mod + '.json')
    exmod = import_file(join(exdir, tname + '.py'))
    m = exmod.create_model()
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=10, ncp=5)
    sim = Simulator(m, package=self.sim_mod)
    if hasattr(m, 'var_input'):
        tsim, profiles = sim.simulate(numpoints=100, varying_inputs=m.var_input)
    else:
        tsim, profiles = sim.simulate(numpoints=100)
    sim.initialize_model()
    results = self._store_results(m, profiles)
    if not os.path.exists(bfile):
        with open(bfile, 'w') as f1:
            json.dump(results, f1)
    with open(bfile, 'r') as f2:
        baseline = json.load(f2)
        self.assertStructuredAlmostEqual(results, baseline, abstol=0.01)