from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class AsynchronousSkippedClass(SkippedClassMixin, TestCase):
    pass