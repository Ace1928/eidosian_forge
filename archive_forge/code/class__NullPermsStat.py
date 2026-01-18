import os
import stat
import sys
from breezy import tests
from breezy.bzr.branch import BzrBranch
from breezy.bzr.remote import RemoteBranchFormat
from breezy.controldir import ControlDir
from breezy.tests.test_permissions import check_mode_r
class _NullPermsStat:
    """A class that proxy's a stat result and strips permissions."""

    def __init__(self, orig_stat):
        self._orig_stat = orig_stat
        self.st_mode = orig_stat.st_mode & ~4095

    def __getattr__(self, name):
        return getattr(self._orig_stat, name)