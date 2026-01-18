from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class AsynchronousDeprecatedReasonlessSkip(DeprecatedReasonlessSkipMixin, TestCase):
    pass