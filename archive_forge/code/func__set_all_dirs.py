import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _set_all_dirs(self, basedir, readonly=True):
    """Recursively set all directories beneath this one."""
    if readonly:
        mode = 365
    else:
        mode = 493
    for root, dirs, files in os.walk(basedir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            os.chmod(path, mode)