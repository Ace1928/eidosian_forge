from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class SkippingMixin:

    def test_skip1(self):
        raise SkipTest('skip1')

    def test_skip2(self):
        raise RuntimeError('I should not get raised')
    test_skip2.skip = 'skip2'

    def test_skip3(self):
        self.fail('I should not fail')
    test_skip3.skip = 'skip3'