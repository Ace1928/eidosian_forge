import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
class TestGitRepository(tests.TestCaseWithTransport):

    def _do_commit(self):
        builder = tests.GitBranchBuilder()
        builder.set_file(b'a', b'text for a\n', False)
        commit_handle = builder.commit(b'Joe Foo <joe@foo.com>', b'message')
        mapping = builder.finish()
        return mapping[commit_handle]

    def setUp(self):
        tests.TestCaseWithTransport.setUp(self)
        dulwich.repo.Repo.create(self.test_dir)
        self.git_repo = Repository.open(self.test_dir)

    def test_supports_rich_root(self):
        repo = self.git_repo
        self.assertEqual(repo.supports_rich_root(), True)

    def test_get_signature_text(self):
        self.assertRaises(errors.NoSuchRevision, self.git_repo.get_signature_text, revision.NULL_REVISION)

    def test_has_signature_for_revision_id(self):
        self.assertEqual(False, self.git_repo.has_signature_for_revision_id(revision.NULL_REVISION))

    def test_all_revision_ids_none(self):
        self.assertEqual([], self.git_repo.all_revision_ids())

    def test_get_known_graph_ancestry(self):
        cid = self._do_commit()
        revid = default_mapping.revision_id_foreign_to_bzr(cid)
        g = self.git_repo.get_known_graph_ancestry([revid])
        self.assertEqual(frozenset([revid]), g.heads([revid]))
        self.assertEqual([(revid, 0, (1,), True)], [(n.key, n.merge_depth, n.revno, n.end_of_merge) for n in g.merge_sort(revid)])

    def test_all_revision_ids(self):
        commit_id = self._do_commit()
        self.assertEqual([default_mapping.revision_id_foreign_to_bzr(commit_id)], self.git_repo.all_revision_ids())

    def assertIsNullInventory(self, inv):
        self.assertEqual(inv.root, None)
        self.assertEqual(inv.revision_id, revision.NULL_REVISION)
        self.assertEqual(list(inv.iter_entries()), [])

    def test_revision_tree_none(self):
        repo = self.git_repo
        tree = repo.revision_tree(revision.NULL_REVISION)
        self.assertEqual(tree.get_revision_id(), revision.NULL_REVISION)

    def test_get_parent_map_null(self):
        self.assertEqual({revision.NULL_REVISION: ()}, self.git_repo.get_parent_map([revision.NULL_REVISION]))