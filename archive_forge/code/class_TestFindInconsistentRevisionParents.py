from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestFindInconsistentRevisionParents(TestCaseWithBrokenRevisionIndex):
    scenarios = all_repository_vf_format_scenarios()

    def test__find_inconsistent_revision_parents(self):
        """_find_inconsistent_revision_parents finds revisions with broken
        parents.
        """
        repo = self.make_repo_with_extra_ghost_index()
        self.assertEqual([(b'revision-id', (b'incorrect-parent',), ())], list(repo._find_inconsistent_revision_parents()))

    def test__check_for_inconsistent_revision_parents(self):
        """_check_for_inconsistent_revision_parents raises BzrCheckError if
        there are any revisions with inconsistent parents.
        """
        repo = self.make_repo_with_extra_ghost_index()
        self.assertRaises(errors.BzrCheckError, repo._check_for_inconsistent_revision_parents)

    def test__check_for_inconsistent_revision_parents_on_clean_repo(self):
        """_check_for_inconsistent_revision_parents does nothing if there are
        no broken revisions.
        """
        repo = self.make_repository('empty-repo')
        if not repo._format.revision_graph_can_have_wrong_parents:
            raise TestNotApplicable('%r cannot have corrupt revision index.' % repo)
        with repo.lock_read():
            repo._check_for_inconsistent_revision_parents()

    def test_check_reports_bad_ancestor(self):
        repo = self.make_repo_with_extra_ghost_index()
        check_object = repo.check(['ignored'])
        check_object.report_results(verbose=False)
        self.assertContainsRe(self.get_log(), '1 revisions have incorrect parents in the revision index')
        check_object.report_results(verbose=True)
        self.assertContainsRe(self.get_log(), 'revision-id has wrong parents in index: \\(incorrect-parent\\) should be \\(\\)')