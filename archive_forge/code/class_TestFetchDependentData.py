import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
class TestFetchDependentData(TestCaseWithInterRepository):

    def test_reference(self):
        from_tree = self.make_branch_and_tree('tree')
        to_repo = self.make_to_repository('to')
        if not from_tree.supports_tree_reference() or not from_tree.branch.repository._format.supports_tree_reference or (not to_repo._format.supports_tree_reference):
            raise TestNotApplicable('Need subtree support.')
        if not to_repo._format.supports_full_versioned_files:
            raise TestNotApplicable('Need full versioned files support.')
        subtree = self.make_branch_and_tree('tree/subtree')
        subtree.commit('subrev 1')
        from_tree.add_reference(subtree)
        tree_rev = from_tree.commit('foo')
        to_repo.fetch(from_tree.branch.repository, tree_rev)
        if from_tree.supports_file_ids:
            file_id = from_tree.path2id('subtree')
            with to_repo.lock_read():
                self.assertEqual({(file_id, tree_rev): ()}, to_repo.texts.get_parent_map([(file_id, tree_rev)]))