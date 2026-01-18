from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
class RetryBackoffExponentialTest(common.HeatTestCase):
    scenarios = [('0_0', dict(attempt=0, scale_factor=0.0, delay=0.0)), ('0_1', dict(attempt=0, scale_factor=1.0, delay=1.0)), ('1_1', dict(attempt=1, scale_factor=1.0, delay=2.0)), ('2_1', dict(attempt=2, scale_factor=1.0, delay=4.0)), ('3_1', dict(attempt=3, scale_factor=1.0, delay=8.0)), ('4_1', dict(attempt=4, scale_factor=1.0, delay=16.0)), ('4_4', dict(attempt=4, scale_factor=4.0, delay=64.0))]

    def test_backoff_delay(self):
        delay = util.retry_backoff_delay(self.attempt, self.scale_factor)
        self.assertEqual(self.delay, delay)