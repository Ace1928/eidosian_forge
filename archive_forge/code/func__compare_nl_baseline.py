import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def _compare_nl_baseline(self, baseline, testfile):
    self.assertEqual(*load_and_compare_nl_baseline(baseline, testfile, self._nl_version))