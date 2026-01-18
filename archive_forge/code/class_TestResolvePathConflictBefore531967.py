import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolvePathConflictBefore531967(TestResolvePathConflict):
    """Same as TestResolvePathConflict but a specific conflict object.
    """

    def assertPathConflict(self, c):
        old_c = bzr_conflicts.PathConflict('<deleted>', self._item_path, file_id=None)
        wt.set_conflicts([old_c])