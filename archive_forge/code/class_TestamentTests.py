import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
class TestamentTests(TestamentSetup):

    def testament_class(self):
        return Testament

    def expected(self, key):
        return texts[self.testament_class()][key]

    def from_revision(self, repository, revision_id):
        return self.testament_class().from_revision(repository, revision_id)

    def test_null_testament(self):
        """Testament for a revision with no contents."""
        t = self.from_revision(self.b.repository, b'test@user-1')
        ass = self.assertTrue
        eq = self.assertEqual
        ass(isinstance(t, Testament))
        eq(t.revision_id, b'test@user-1')
        eq(t.committer, 'test@user')
        eq(t.timestamp, 1129025423)
        eq(t.timezone, 0)

    def test_testment_text_form(self):
        """Conversion of testament to canonical text form."""
        t = self.from_revision(self.b.repository, b'test@user-1')
        text_form = t.as_text()
        self.log('testament text form:\n%s' % text_form)
        self.assertEqualDiff(text_form, self.expected('rev_1'))
        short_text_form = t.as_short_text()
        self.assertEqualDiff(short_text_form, self.expected('rev_1_short'))

    def test_testament_with_contents(self):
        """Testament containing a file and a directory."""
        t = self.from_revision(self.b.repository, b'test@user-2')
        text_form = t.as_text()
        self.log('testament text form:\n%s' % text_form)
        self.assertEqualDiff(text_form, self.expected('rev_2'))
        actual_short = t.as_short_text()
        self.assertEqualDiff(actual_short, self.expected('rev_2_short'))

    def test_testament_symlinks(self):
        """Testament containing symlink (where possible)"""
        self.requireFeature(SymlinkFeature(self.test_dir))
        os.symlink('wibble/linktarget', 'link')
        self.wt.add(['link'], ids=[b'link-id'])
        self.wt.commit(message='add symlink', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='test@user')
        t = self.from_revision(self.b.repository, b'test@user-3')
        self.assertEqualDiff(t.as_text(), self.expected('rev_3'))

    def test_testament_revprops(self):
        """Testament to revision with extra properties"""
        props = {'flavor': 'sour cherry\ncream cheese', 'size': 'medium', 'empty': ''}
        self.wt.commit(message='revision with properties', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='test@user', revprops=props)
        t = self.from_revision(self.b.repository, b'test@user-3')
        self.assertEqualDiff(t.as_text(), self.expected('rev_props'))

    def test_testament_unicode_commit_message(self):
        self.wt.commit(message='non-ascii commit © me', timestamp=1129025493, timezone=36000, rev_id=b'test@user-3', committer='Erik Bågfors <test@user>', revprops={'uni': 'µ'})
        t = self.from_revision(self.b.repository, b'test@user-3')
        self.assertEqualDiff(self.expected('sample_unicode').encode('utf-8'), t.as_text())

    def test_from_tree(self):
        tree = self.b.repository.revision_tree(b'test@user-2')
        testament = self.testament_class().from_revision_tree(tree)
        text_1 = testament.as_short_text()
        text_2 = self.from_revision(self.b.repository, b'test@user-2').as_short_text()
        self.assertEqual(text_1, text_2)

    def test___init__(self):
        revision = self.b.repository.get_revision(b'test@user-2')
        tree = self.b.repository.revision_tree(b'test@user-2')
        testament_1 = self.testament_class()(revision, tree)
        text_1 = testament_1.as_short_text()
        text_2 = self.from_revision(self.b.repository, b'test@user-2').as_short_text()
        self.assertEqual(text_1, text_2)