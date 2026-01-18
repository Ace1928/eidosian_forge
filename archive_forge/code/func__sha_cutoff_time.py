import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def _sha_cutoff_time(self):
    timestamp = super()._sha_cutoff_time()
    self._cutoff_time = timestamp + self._time_offset