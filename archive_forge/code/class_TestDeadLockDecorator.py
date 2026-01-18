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
class TestDeadLockDecorator(_base.BaseTestCase):

    @db_api.retry_db_errors
    def _decorated_function(self, fail_count, exc_to_raise):
        self.fail_count = getattr(self, 'fail_count', fail_count + 1) - 1
        if self.fail_count:
            raise exc_to_raise

    def test_regular_exception_excluded(self):
        with testtools.ExpectedException(ValueError):
            self._decorated_function(1, ValueError)

    def test_staledata_error_caught(self):
        e = exc.StaleDataError()
        self.assertIsNone(self._decorated_function(1, e))

    def test_dbconnection_error_caught(self):
        e = db_exc.DBConnectionError()
        self.assertIsNone(self._decorated_function(1, e))

    def test_multi_exception_contains_retry(self):
        e = exceptions.MultipleExceptions([ValueError(), db_exc.RetryRequest(TypeError())])
        self.assertIsNone(self._decorated_function(1, e))

    def test_multi_exception_contains_deadlock(self):
        e = exceptions.MultipleExceptions([ValueError(), db_exc.DBDeadlock()])
        self.assertIsNone(self._decorated_function(1, e))

    def test_multi_nested_exception_contains_deadlock(self):
        i = exceptions.MultipleExceptions([ValueError(), db_exc.DBDeadlock()])
        e = exceptions.MultipleExceptions([ValueError(), i])
        self.assertIsNone(self._decorated_function(1, e))

    def test_multi_exception_raised_on_exceed(self):
        retry_fixture = fixture.DBRetryErrorsFixture(max_retries=2)
        retry_fixture.setUp()
        e = exceptions.MultipleExceptions([ValueError(), db_exc.DBDeadlock()])
        with testtools.ExpectedException(exceptions.MultipleExceptions):
            self._decorated_function(db_api.MAX_RETRIES + 1, e)
        retry_fixture.cleanUp()

    def test_mysql_savepoint_error(self):
        e = db_exc.DBError("(pymysql.err.InternalError) (1305, 'SAVEPOINT sa_savepoint_1 does not exist')")
        self.assertIsNone(self._decorated_function(1, e))

    @db_api.retry_if_session_inactive('alt_context')
    def _alt_context_function(self, alt_context, *args, **kwargs):
        return self._decorated_function(*args, **kwargs)

    @db_api.retry_if_session_inactive()
    def _context_function(self, context, list_arg, dict_arg, fail_count, exc_to_raise):
        list_arg.append(1)
        dict_arg[max(dict_arg.keys()) + 1] = True
        self.fail_count = getattr(self, 'fail_count', fail_count + 1) - 1
        if self.fail_count:
            raise exc_to_raise
        return (list_arg, dict_arg)

    def test_stacked_retries_dont_explode_retry_count(self):
        context = mock.Mock()
        context.session.is_active = False
        e = db_exc.DBConnectionError()
        mock.patch('time.sleep').start()
        with testtools.ExpectedException(db_exc.DBConnectionError):
            self._alt_context_function(context, db_api.MAX_RETRIES + 1, e)

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

    def test_all_deadlock_time_elapsed(self):
        self._test_retry_time_cost(db_exc.DBDeadlock)

    def test_not_deadlock_time_elapsed(self):
        self._test_retry_time_cost(db_exc.DBConnectionError)

    def test_retry_if_session_inactive_args_not_mutated_after_retries(self):
        context = mock.Mock()
        context.session.is_active = False
        list_arg = [1, 2, 3, 4]
        dict_arg = {1: 'a', 2: 'b'}
        l, d = self._context_function(context, list_arg, dict_arg, 5, db_exc.DBDeadlock())
        self.assertEqual(5, len(l))
        self.assertEqual(3, len(d))

    def test_retry_if_session_inactive_kwargs_not_mutated_after_retries(self):
        context = mock.Mock()
        context.session.is_active = False
        list_arg = [1, 2, 3, 4]
        dict_arg = {1: 'a', 2: 'b'}
        l, d = self._context_function(context, list_arg=list_arg, dict_arg=dict_arg, fail_count=5, exc_to_raise=db_exc.DBDeadlock())
        self.assertEqual(5, len(l))
        self.assertEqual(3, len(d))

    def test_retry_if_session_inactive_no_retry_in_active_session(self):
        context = mock.Mock()
        context.session.is_active = True
        with testtools.ExpectedException(db_exc.DBDeadlock):
            self._context_function(context, [], {1: 2}, fail_count=1, exc_to_raise=db_exc.DBDeadlock())