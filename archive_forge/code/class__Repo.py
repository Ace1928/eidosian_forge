import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class _Repo:
    """A minimal api to get InventoryRevisionTree to work."""

    def __init__(self):
        default_format = controldir.format_registry.make_controldir('default')
        self._format = default_format.repository_format

    def lock_read(self):
        pass

    def unlock(self):
        pass