import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
class ExceptionFileMover(_FileMover):

    def __init__(self, bad_source=None, bad_target=None):
        _FileMover.__init__(self)
        self.bad_source = bad_source
        self.bad_target = bad_target

    def rename(self, source, target):
        if self.bad_source is not None and source.endswith(self.bad_source):
            raise Bogus
        elif self.bad_target is not None and target.endswith(self.bad_target):
            raise Bogus
        else:
            _FileMover.rename(self, source, target)