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
class TestDevelopment6FindParentIdsOfRevisions(TestCaseWithTransport):
    """Tests for _find_parent_ids_of_revisions."""

    def setUp(self):
        super().setUp()
        self.builder = self.make_branch_builder('source')
        self.builder.start_series()
        self.builder.build_snapshot(None, [('add', ('', b'tree-root', 'directory', None))], revision_id=b'initial')
        self.repo = self.builder.get_branch().repository
        self.addCleanup(self.builder.finish_series)

    def assertParentIds(self, expected_result, rev_set):
        self.assertEqual(sorted(expected_result), sorted(self.repo._find_parent_ids_of_revisions(rev_set)))

    def test_simple(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        self.builder.build_snapshot([b'revid1'], [], revision_id=b'revid2')
        rev_set = [b'revid2']
        self.assertParentIds([b'revid1'], rev_set)

    def test_not_first_parent(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        self.builder.build_snapshot([b'revid1'], [], revision_id=b'revid2')
        self.builder.build_snapshot([b'revid2'], [], revision_id=b'revid3')
        rev_set = [b'revid3', b'revid2']
        self.assertParentIds([b'revid1'], rev_set)

    def test_not_null(self):
        rev_set = [b'initial']
        self.assertParentIds([], rev_set)

    def test_not_null_set(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        rev_set = [_mod_revision.NULL_REVISION]
        self.assertParentIds([], rev_set)

    def test_ghost(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        rev_set = [b'ghost', b'revid1']
        self.assertParentIds([b'initial'], rev_set)

    def test_ghost_parent(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        self.builder.build_snapshot([b'revid1', b'ghost'], [], revision_id=b'revid2')
        rev_set = [b'revid2', b'revid1']
        self.assertParentIds([b'ghost', b'initial'], rev_set)

    def test_righthand_parent(self):
        self.builder.build_snapshot(None, [], revision_id=b'revid1')
        self.builder.build_snapshot([b'revid1'], [], revision_id=b'revid2a')
        self.builder.build_snapshot([b'revid1'], [], revision_id=b'revid2b')
        self.builder.build_snapshot([b'revid2a', b'revid2b'], [], revision_id=b'revid3')
        rev_set = [b'revid3', b'revid2a']
        self.assertParentIds([b'revid1', b'revid2b'], rev_set)