from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousSkippedClass(SkippedClassMixin, SynchronousTestCase):
    pass