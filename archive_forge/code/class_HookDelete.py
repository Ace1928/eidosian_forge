import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
class HookDelete(_mod_merge.AbstractPerFileMerger):

    def merge_contents(self, merge_params):
        test.hook_log.append(('delete',))
        if merge_params.this_path == 'name1':
            return ('delete', None)
        return ('not_applicable', None)