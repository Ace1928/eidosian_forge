import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def _sha1_file_and_log(self, abspath):
    self._log.append(('sha1', abspath))
    return self._sha1_provider.sha1(abspath)