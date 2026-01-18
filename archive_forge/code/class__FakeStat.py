import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class _FakeStat:
    """A class with the same attributes as a real stat result."""

    def __init__(self, size, mtime, ctime, dev, ino, mode):
        self.st_size = size
        self.st_mtime = mtime
        self.st_ctime = ctime
        self.st_dev = dev
        self.st_ino = ino
        self.st_mode = mode

    @staticmethod
    def from_stat(st):
        return _FakeStat(st.st_size, st.st_mtime, st.st_ctime, st.st_dev, st.st_ino, st.st_mode)