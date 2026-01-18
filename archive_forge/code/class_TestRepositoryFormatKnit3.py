from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestRepositoryFormatKnit3(TestCaseWithTransport):

    def test_attribute__fetch_order(self):
        """Knits need topological data insertion."""
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit3()
        repo = self.make_repository('.', format=format)
        self.assertEqual('topological', repo._format._fetch_order)

    def test_attribute__fetch_uses_deltas(self):
        """Knits reuse deltas."""
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit3()
        repo = self.make_repository('.', format=format)
        self.assertEqual(True, repo._format._fetch_uses_deltas)

    def test_convert(self):
        """Ensure the upgrade adds weaves for roots"""
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit1()
        tree = self.make_branch_and_tree('.', format)
        tree.commit('Dull commit', rev_id=b'dull')
        revision_tree = tree.branch.repository.revision_tree(b'dull')
        with revision_tree.lock_read():
            self.assertRaises(transport.NoSuchFile, revision_tree.get_file_lines, '')
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit3()
        upgrade.Convert('.', format)
        tree = workingtree.WorkingTree.open('.')
        revision_tree = tree.branch.repository.revision_tree(b'dull')
        with revision_tree.lock_read():
            revision_tree.get_file_lines('')
        tree.commit('Another dull commit', rev_id=b'dull2')
        revision_tree = tree.branch.repository.revision_tree(b'dull2')
        revision_tree.lock_read()
        self.addCleanup(revision_tree.unlock)
        self.assertEqual(b'dull', revision_tree.get_file_revision(''))

    def test_supports_external_lookups(self):
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit3()
        repo = self.make_repository('.', format=format)
        self.assertFalse(repo._format.supports_external_lookups)