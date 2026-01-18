from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestFindTextKeyReferences(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def test_empty(self):
        repo = self.make_repository('.')
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({}, repo.find_text_key_references())