from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
class RetryBackoffJitterTest(common.HeatTestCase):
    scenarios = [('0_0_1', dict(attempt=0, scale_factor=0.0, jitter_max=1.0, delay_from=0.0, delay_to=1.0)), ('1_1_1', dict(attempt=1, scale_factor=1.0, jitter_max=1.0, delay_from=2.0, delay_to=3.0)), ('1_1_5', dict(attempt=1, scale_factor=1.0, jitter_max=5.0, delay_from=2.0, delay_to=7.0))]

    def test_backoff_delay(self):
        for _ in range(100):
            delay = util.retry_backoff_delay(self.attempt, self.scale_factor, self.jitter_max)
            self.assertThat(delay, matchers.GreaterThan(self.delay_from))
            self.assertThat(delay, matchers.LessThan(self.delay_to))