import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestCompiledLtPathByDirblock(TestLtPathByDirblock):
    """Test the pyrex implementation of _lt_path_by_dirblock"""
    _test_needs_features = [compiled_dirstate_helpers_feature]

    def get_lt_path_by_dirblock(self):
        from .._dirstate_helpers_pyx import _lt_path_by_dirblock
        return _lt_path_by_dirblock