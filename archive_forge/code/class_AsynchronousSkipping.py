from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class AsynchronousSkipping(SkippingMixin, TestCase):
    pass