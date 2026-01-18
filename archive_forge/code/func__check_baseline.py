import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def _check_baseline(self, model, **kwds):
    baseline_fname, test_fname = self._get_fnames()
    self._cleanup(test_fname)
    io_options = {'symbolic_solver_labels': True, 'output_fixed_variables': True}
    io_options.update(kwds)
    model.write(test_fname, format='gams', io_options=io_options)
    try:
        self.assertTrue(cmp(test_fname, baseline_fname))
    except:
        with open(baseline_fname, 'r') as f1, open(test_fname, 'r') as f2:
            f1_contents = list(filter(None, f1.read().split()))
            f2_contents = list(filter(None, f2.read().split()))
            self.assertEqual(f1_contents, f2_contents, '\n\nbaseline: %s\ntestFile: %s\n' % (baseline_fname, test_fname))
    self._cleanup(test_fname)