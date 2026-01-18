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
def _prepare_orphan(self, wt):
    self.build_tree(['dir/', 'dir/file', 'dir/foo'])
    wt.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
    wt.commit('add dir and file ignoring foo')
    tt = wt.transform()
    self.addCleanup(tt.finalize)
    dir_tid = tt.trans_id_tree_path('dir')
    file_tid = tt.trans_id_tree_path('dir/file')
    orphan_tid = tt.trans_id_tree_path('dir/foo')
    tt.delete_contents(file_tid)
    tt.unversion_file(file_tid)
    tt.delete_contents(dir_tid)
    tt.unversion_file(dir_tid)
    raw_conflicts = tt.find_raw_conflicts()
    self.assertLength(1, raw_conflicts)
    self.assertEqual(('missing parent', 'new-1'), raw_conflicts[0])
    return (tt, orphan_tid)