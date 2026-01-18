from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
def _test_retry_time_cost(self, exc_to_raise):
    worst_case = [0.5, 1, 2, 4, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    class FakeTime(object):

        def __init__(self):
            self.counter = 0

        def sleep(self, t):
            self.counter += t
    fake_timer = FakeTime()

    def fake_sleep(t):
        fake_timer.sleep(t)
    e = exc_to_raise()
    mock.patch('time.sleep', side_effect=fake_sleep).start()
    with testtools.ExpectedException(exc_to_raise):
        self._decorated_function(db_api.MAX_RETRIES + 1, e)
    if exc_to_raise == db_exc.DBDeadlock:
        self.assertEqual(True, fake_timer.counter <= sum(worst_case))
    else:
        self.assertGreaterEqual(sum(worst_case), fake_timer.counter)