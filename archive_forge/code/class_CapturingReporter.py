import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
class CapturingReporter(NullCommitReporter):
    """This reporter captures the calls made to it for evaluation later."""

    def __init__(self):
        self.calls = []

    def snapshot_change(self, change, path):
        self.calls.append(('change', change, path))

    def deleted(self, file_id):
        self.calls.append(('deleted', file_id))

    def missing(self, path):
        self.calls.append(('missing', path))

    def renamed(self, change, old_path, new_path):
        self.calls.append(('renamed', change, old_path, new_path))

    def is_verbose(self):
        return True