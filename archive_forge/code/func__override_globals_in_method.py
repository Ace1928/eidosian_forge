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
def _override_globals_in_method(self, instance, method_name, globals):
    """Replace method on instance with one with updated globals"""
    import types
    func = getattr(instance, method_name).__func__
    new_globals = dict(func.__globals__)
    new_globals.update(globals)
    new_func = types.FunctionType(func.__code__, new_globals, func.__name__, func.__defaults__)
    setattr(instance, method_name, types.MethodType(new_func, instance))
    self.addCleanup(delattr, instance, method_name)