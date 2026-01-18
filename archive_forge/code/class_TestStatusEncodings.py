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
class TestStatusEncodings(TestCaseWithTransport):

    def make_uncommitted_tree(self):
        """Build a branch with uncommitted unicode named changes in the cwd."""
        working_tree = self.make_branch_and_tree('.')
        filename = 'hellØ'
        try:
            self.build_tree_contents([(filename, b'contents of hello')])
        except UnicodeEncodeError:
            raise TestSkipped("can't build unicode working tree in filesystem encoding %s" % sys.getfilesystemencoding())
        working_tree.add(filename)
        return working_tree

    def test_stdout_ascii(self):
        self.overrideAttr(osutils, '_cached_user_encoding', 'ascii')
        working_tree = self.make_uncommitted_tree()
        stdout, stderr = self.run_bzr('status')
        self.assertEqual(stdout, 'added:\n  hell?\n')

    def test_stdout_latin1(self):
        self.overrideAttr(osutils, '_cached_user_encoding', 'latin-1')
        working_tree = self.make_uncommitted_tree()
        stdout, stderr = self.run_bzr('status')
        expected = 'added:\n  hellØ\n'
        self.assertEqual(stdout, expected)