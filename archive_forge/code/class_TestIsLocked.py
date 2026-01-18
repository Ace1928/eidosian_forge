from breezy.tests.per_repository import TestCaseWithRepository
class TestIsLocked(TestCaseWithRepository):

    def test_not_locked(self):
        repo = self.make_repository('.')
        self.assertFalse(repo.is_locked())

    def test_read_locked(self):
        repo = self.make_repository('.')
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertTrue(repo.is_locked())

    def test_write_locked(self):
        repo = self.make_repository('.')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        self.assertTrue(repo.is_locked())