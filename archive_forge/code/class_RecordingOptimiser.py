from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
class RecordingOptimiser(InterTree):
    calls: List[Tuple[str, ...]] = []

    def compare(self, want_unchanged=False, specific_files=None, extra_trees=None, require_versioned=False, include_root=False, want_unversioned=False):
        self.calls.append(('compare', self.source, self.target, want_unchanged, specific_files, extra_trees, require_versioned, include_root, want_unversioned))

    def find_source_path(self, target_path, recurse='none'):
        self.calls.append(('find_source_path', self.source, self.target, target_path, recurse))

    @classmethod
    def is_compatible(klass, source, target):
        return True