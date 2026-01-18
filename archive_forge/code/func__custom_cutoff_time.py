import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _custom_cutoff_time(self):
    """We need to fake the cutoff time."""
    return time.time() + 10.0