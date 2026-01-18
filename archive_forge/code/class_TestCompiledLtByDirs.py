import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestCompiledLtByDirs(TestLtByDirs):
    """Test the pyrex implementation of lt_by_dirs"""
    _test_needs_features = [compiled_dirstate_helpers_feature]

    def get_lt_by_dirs(self):
        from .._dirstate_helpers_pyx import lt_by_dirs
        return lt_by_dirs