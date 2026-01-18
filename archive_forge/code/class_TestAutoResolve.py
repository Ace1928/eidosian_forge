import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
class TestAutoResolve(TestCaseWithTransport):

    def test_auto_resolve(self):
        base = self.make_branch_and_tree('base')
        self.build_tree_contents([('base/hello', b'Hello')])
        base.add('hello', ids=b'hello_id')
        base.commit('Hello')
        other = base.controldir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/hello', b'hELLO')])
        other.commit('Case switch')
        this = base.controldir.sprout('this').open_workingtree()
        self.assertPathExists('this/hello')
        self.build_tree_contents([('this/hello', b'Hello World')])
        this.commit('Add World')
        this.merge_from_branch(other.branch)
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], this.conflicts())
        this.auto_resolve()
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], this.conflicts())
        self.build_tree_contents([('this/hello', '<<<<<<<')])
        this.auto_resolve()
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], this.conflicts())
        self.build_tree_contents([('this/hello', '=======')])
        this.auto_resolve()
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], this.conflicts())
        self.build_tree_contents([('this/hello', '\n>>>>>>>')])
        remaining, resolved = this.auto_resolve()
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], this.conflicts())
        self.assertEqual([], resolved)
        self.build_tree_contents([('this/hello', b'hELLO wORLD')])
        remaining, resolved = this.auto_resolve()
        self.assertEqual([], this.conflicts())
        self.assertEqual([_mod_bzr_conflicts.TextConflict('hello', b'hello_id')], resolved)
        self.assertPathDoesNotExist('this/hello.BASE')

    def test_unsupported_symlink_auto_resolve(self):
        self.requireFeature(SymlinkFeature(self.test_dir))
        base = self.make_branch_and_tree('base')
        self.build_tree_contents([('base/hello', 'Hello')])
        base.add('hello', ids=b'hello_id')
        base.commit('commit 0')
        other = base.controldir.sprout('other').open_workingtree()
        self.build_tree_contents([('other/hello', 'Hello')])
        os.symlink('other/hello', 'other/foo')
        other.add('foo', ids=b'foo_id')
        other.commit('commit symlink')
        this = base.controldir.sprout('this').open_workingtree()
        self.assertPathExists('this/hello')
        self.build_tree_contents([('this/hello', 'Hello')])
        this.commit('commit 2')
        log = BytesIO()
        trace.push_log_file(log)
        os_symlink = getattr(os, 'symlink', None)
        os.symlink = None
        try:
            this.merge_from_branch(other.branch)
        finally:
            if os_symlink:
                os.symlink = os_symlink
        self.assertContainsRe(log.getvalue(), b'Unable to create symlink "foo" on this filesystem')

    def test_auto_resolve_dir(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/hello/'])
        tree.add('hello', ids=b'hello-id')
        file_conflict = _mod_bzr_conflicts.TextConflict('hello', b'hello-id')
        tree.set_conflicts([file_conflict])
        remaining, resolved = tree.auto_resolve()
        self.assertEqual(remaining, conflicts.ConflictList([_mod_bzr_conflicts.TextConflict('hello', 'hello-id')]))
        self.assertEqual(resolved, [])

    def test_auto_resolve_missing(self):
        tree = self.make_branch_and_tree('tree')
        file_conflict = _mod_bzr_conflicts.TextConflict('hello', b'hello-id')
        tree.set_conflicts([file_conflict])
        remaining, resolved = tree.auto_resolve()
        self.assertEqual(remaining, [])
        self.assertEqual(resolved, conflicts.ConflictList([_mod_bzr_conflicts.TextConflict('hello', 'hello-id')]))