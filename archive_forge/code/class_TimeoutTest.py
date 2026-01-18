import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class TimeoutTest(common.HeatTestCase):

    def test_compare(self):
        task = scheduler.TaskRunner(DummyTask())
        earlier = scheduler.Timeout(task, 10)
        eventlet.sleep(0.01)
        later = scheduler.Timeout(task, 10)
        self.assertTrue(earlier.earlier_than(later))
        self.assertFalse(later.earlier_than(earlier))