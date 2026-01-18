from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SynchronousSkipping(SkippingMixin, SynchronousTestCase):
    pass