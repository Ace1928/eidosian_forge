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
class TestOrphan(tests.TestCaseWithTransport):

    def test_no_orphan_for_transform_preview(self):
        tree = self.make_branch_and_tree('tree')
        tt = tree.preview_transform()
        self.addCleanup(tt.finalize)
        self.assertRaises(NotImplementedError, tt.new_orphan, 'foo', 'bar')

    def _set_orphan_policy(self, wt, policy):
        wt.branch.get_config_stack().set('transform.orphan_policy', policy)

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

    def test_new_orphan_created(self):
        wt = self.make_branch_and_tree('.')
        self._set_orphan_policy(wt, 'move')
        tt, orphan_tid = self._prepare_orphan(wt)
        warnings = []

        def warning(*args):
            warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)
        remaining_conflicts = resolve_conflicts(tt)
        self.assertEqual(['dir/foo has been orphaned in brz-orphans'], warnings)
        self.assertLength(0, remaining_conflicts)
        self.assertEqual('foo.~1~', tt.final_name(orphan_tid))
        self.assertEqual('brz-orphans', tt.final_name(tt.final_parent(orphan_tid)))

    def test_never_orphan(self):
        wt = self.make_branch_and_tree('.')
        self._set_orphan_policy(wt, 'conflict')
        tt, orphan_tid = self._prepare_orphan(wt)
        remaining_conflicts = resolve_conflicts(tt)
        self.assertLength(1, remaining_conflicts)
        self.assertEqual(('deleting parent', 'Not deleting', 'new-1'), remaining_conflicts.pop())

    def test_orphan_error(self):

        def bogus_orphan(tt, orphan_id, parent_id):
            raise transform.OrphaningError(tt.final_name(orphan_id), tt.final_name(parent_id))
        transform.orphaning_registry.register('bogus', bogus_orphan, 'Raise an error when orphaning')
        wt = self.make_branch_and_tree('.')
        self._set_orphan_policy(wt, 'bogus')
        tt, orphan_tid = self._prepare_orphan(wt)
        remaining_conflicts = resolve_conflicts(tt)
        self.assertLength(1, remaining_conflicts)
        self.assertEqual(('deleting parent', 'Not deleting', 'new-1'), remaining_conflicts.pop())

    def test_unknown_orphan_policy(self):
        wt = self.make_branch_and_tree('.')
        self._set_orphan_policy(wt, 'donttouchmypreciouuus')
        tt, orphan_tid = self._prepare_orphan(wt)
        warnings = []

        def warning(*args):
            warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)
        remaining_conflicts = resolve_conflicts(tt)
        self.assertLength(1, remaining_conflicts)
        self.assertEqual(('deleting parent', 'Not deleting', 'new-1'), remaining_conflicts.pop())
        self.assertLength(1, warnings)
        self.assertStartsWith(warnings[0], 'Value "donttouchmypreciouuus" ')