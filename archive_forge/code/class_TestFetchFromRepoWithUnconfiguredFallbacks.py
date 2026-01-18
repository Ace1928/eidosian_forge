from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
class TestFetchFromRepoWithUnconfiguredFallbacks(TestFetchBase):

    def make_stacked_source_repo(self):
        _, source_b = self.make_source_branch()
        stack_b = self.make_branch('stack-on')
        stack_b.pull(source_b, stop_revision=b'B-id')
        stacked_b = self.make_branch('stacked')
        stacked_b.set_stacked_on_url('../stack-on')
        stacked_b.pull(source_b, stop_revision=b'C-id')
        return stacked_b.repository

    def test_fetch_everything_includes_parent_invs(self):
        stacked = self.make_stacked_source_repo()
        repo_missing_fallbacks = stacked.controldir.open_repository()
        self.addCleanup(repo_missing_fallbacks.lock_read().unlock)
        target = self.make_repository('target')
        self.addCleanup(target.lock_write().unlock)
        target.fetch(repo_missing_fallbacks, fetch_spec=vf_search.EverythingResult(repo_missing_fallbacks))
        self.assertEqual(repo_missing_fallbacks.revisions.keys(), target.revisions.keys())
        self.assertEqual(repo_missing_fallbacks.inventories.keys(), target.inventories.keys())
        self.assertEqual([b'C-id'], sorted((k[-1] for k in target.revisions.keys())))
        self.assertEqual([b'B-id', b'C-id'], sorted((k[-1] for k in target.inventories.keys())))