import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
class TestStatus(TestCaseWithTransport):

    def test_status_plain(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        result = self.run_bzr('status')[0]
        self.assertContainsRe(result, 'unknown:\n  hello.txt\n')
        tree.add('hello.txt')
        result = self.run_bzr('status')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        tree.commit(message='added')
        result = self.run_bzr('status -r 0..1')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        result = self.run_bzr('status -c 1')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        self.build_tree(['world.txt'])
        result = self.run_bzr('status -r 0')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\nunknown:\n  world.txt\n')
        result2 = self.run_bzr('status -r 0..')[0]
        self.assertEqual(result2, result)

    def test_status_short(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        result = self.run_bzr('status --short')[0]
        self.assertContainsRe(result, '[?]   hello.txt\n')
        tree.add('hello.txt')
        result = self.run_bzr('status --short')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n')
        tree.commit(message='added')
        result = self.run_bzr('status --short -r 0..1')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n')
        self.build_tree(['world.txt'])
        result = self.run_bzr('status -S -r 0')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n[?]   world.txt\n')
        result2 = self.run_bzr('status -S -r 0..')[0]
        self.assertEqual(result2, result)

    def test_status_versioned(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        result = self.run_bzr('status --versioned')[0]
        self.assertNotContainsRe(result, 'unknown:\n  hello.txt\n')
        tree.add('hello.txt')
        result = self.run_bzr('status --versioned')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        tree.commit('added')
        result = self.run_bzr('status --versioned -r 0..1')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        self.build_tree(['world.txt'])
        result = self.run_bzr('status --versioned -r 0')[0]
        self.assertContainsRe(result, 'added:\n  hello.txt\n')
        self.assertNotContainsRe(result, 'unknown:\n  world.txt\n')
        result2 = self.run_bzr('status --versioned -r 0..')[0]
        self.assertEqual(result2, result)

    def test_status_SV(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        result = self.run_bzr('status -SV')[0]
        self.assertNotContainsRe(result, 'hello.txt')
        tree.add('hello.txt')
        result = self.run_bzr('status -SV')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n')
        tree.commit(message='added')
        result = self.run_bzr('status -SV -r 0..1')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n')
        self.build_tree(['world.txt'])
        result = self.run_bzr('status -SV -r 0')[0]
        self.assertContainsRe(result, '[+]N  hello.txt\n')
        result2 = self.run_bzr('status -SV -r 0..')[0]
        self.assertEqual(result2, result)

    def assertStatusContains(self, pattern, short=False):
        """Run status, and assert it contains the given pattern"""
        if short:
            result = self.run_bzr('status --short')[0]
        else:
            result = self.run_bzr('status')[0]
        self.assertContainsRe(result, pattern)

    def test_kind_change_plain(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file'])
        tree.add('file')
        tree.commit('added file')
        unlink('file')
        self.build_tree(['file/'])
        self.assertStatusContains('kind changed:\n  file \\(file => directory\\)')
        tree.rename_one('file', 'directory')
        self.assertStatusContains('renamed:\n  file => directory/\nmodified:\n  directory/\n')
        rmdir('directory')
        self.assertStatusContains('removed:\n  file\n')

    def test_kind_change_short(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['file'])
        tree.add('file')
        tree.commit('added file')
        unlink('file')
        self.build_tree(['file/'])
        self.assertStatusContains('K  file => file/', short=True)
        tree.rename_one('file', 'directory')
        self.assertStatusContains('RK  file => directory/', short=True)
        rmdir('directory')
        self.assertStatusContains('RD  file => directory', short=True)

    def test_status_illegal_revision_specifiers(self):
        out, err = self.run_bzr('status -r 1..23..123', retcode=3)
        self.assertContainsRe(err, 'one or two revision specifiers')

    def test_status_no_pending(self):
        a_tree = self.make_branch_and_tree('a')
        self.build_tree(['a/a'])
        a_tree.add('a')
        a_tree.commit('a')
        b_tree = a_tree.controldir.sprout('b').open_workingtree()
        self.build_tree(['b/b'])
        b_tree.add('b')
        b_tree.commit('b')
        self.run_bzr('merge ../b', working_dir='a')
        out, err = self.run_bzr('status --no-pending', working_dir='a')
        self.assertEqual(out, 'added:\n  b\n')

    def test_pending_specific_files(self):
        """With a specific file list, pending merges are not shown."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a', b'content of a\n')])
        tree.add('a')
        r1_id = tree.commit('one')
        alt = tree.controldir.sprout('alt').open_workingtree()
        self.build_tree_contents([('alt/a', b'content of a\nfrom alt\n')])
        alt_id = alt.commit('alt')
        tree.merge_from_branch(alt.branch)
        output = self.make_utf8_encoded_stringio()
        show_tree_status(tree, to_file=output)
        self.assertContainsRe(output.getvalue(), b'pending merge')
        out, err = self.run_bzr('status tree/a')
        self.assertNotContainsRe(out, 'pending merge')