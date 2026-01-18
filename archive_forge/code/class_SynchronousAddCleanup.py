from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousAddCleanup(AddCleanupMixin, SynchronousTestCase):
    pass