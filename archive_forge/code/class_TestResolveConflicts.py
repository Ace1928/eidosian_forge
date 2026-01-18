import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveConflicts(script.TestCaseWithTransportAndScript):
    preamble: str

    def setUp(self):
        super().setUp()
        self.run_script(self.preamble)