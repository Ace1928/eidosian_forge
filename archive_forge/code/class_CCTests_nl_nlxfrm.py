import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
class CCTests_nl_nlxfrm(CCTests):

    def _test(self, tname, M):
        bfile = os.path.join(currdir, tname + '_nlxfrm.nl')
        xfrm = TransformationFactory('mpec.nl')
        xfrm.apply_to(M)
        fd = FileDeterminism.SORT_INDICES if self._nl_version == 'nl_v2' else 1
        with TempfileManager:
            ofile = TempfileManager.create_tempfile(suffix='_nlxfrm.out')
            M.write(ofile, format=self._nl_version, io_options={'symbolic_solver_labels': False, 'file_determinism': fd})
            self.assertEqual(*load_and_compare_nl_baseline(bfile, ofile, self._nl_version))