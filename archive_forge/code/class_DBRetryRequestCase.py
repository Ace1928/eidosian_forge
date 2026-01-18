from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
class DBRetryRequestCase(DBAPITestCase):

    def test_retry_wrapper_succeeds(self):

        @api.wrap_db_retry(max_retries=10)
        def some_method():
            pass
        some_method()

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_wrapper_reaches_limit(self, mock_sleep):
        max_retries = 2

        @api.wrap_db_retry(max_retries=max_retries)
        def some_method(res):
            res['result'] += 1
            raise exception.RetryRequest(ValueError())
        res = {'result': 0}
        self.assertRaises(ValueError, some_method, res)
        self.assertEqual(max_retries + 1, res['result'])

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_wrapper_exception_checker(self, mock_sleep):

        def exception_checker(exc):
            return isinstance(exc, ValueError) and exc.args[0] < 5

        @api.wrap_db_retry(max_retries=10, exception_checker=exception_checker)
        def some_method(res):
            res['result'] += 1
            raise ValueError(res['result'])
        res = {'result': 0}
        self.assertRaises(ValueError, some_method, res)
        self.assertEqual(5, res['result'])

    @mock.patch.object(DBAPI, 'api_class_call1')
    @mock.patch.object(api, 'wrap_db_retry')
    def test_mocked_methods_are_not_wrapped(self, mocked_wrap, mocked_method):
        dbapi = api.DBAPI('oslo_db.tests.test_api')
        dbapi.api_class_call1()
        self.assertFalse(mocked_wrap.called)

    @mock.patch('oslo_db.api.LOG')
    def test_retry_wrapper_non_db_error_not_logged(self, mock_log):

        @api.wrap_db_retry(max_retries=5, retry_on_deadlock=True)
        def some_method():
            raise AttributeError('test')
        self.assertRaises(AttributeError, some_method)
        self.assertFalse(mock_log.called)

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_wrapper_deadlock(self, mock_sleep):

        @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
        def some_method_no_deadlock():
            raise exception.RetryRequest(ValueError())
        with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
            mock_get.return_value = (2, 2)
            self.assertRaises(ValueError, some_method_no_deadlock)
            mock_get.assert_called_once_with(1, False)

        @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
        def some_method_deadlock():
            raise exception.DBDeadlock('test')
        with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
            mock_get.return_value = (0.1, 2)
            self.assertRaises(exception.DBDeadlock, some_method_deadlock)
            mock_get.assert_called_once_with(1, True)

        @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True, jitter=True)
        def some_method_no_deadlock_exp():
            raise exception.RetryRequest(ValueError())
        with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
            mock_get.return_value = (0.1, 2)
            self.assertRaises(ValueError, some_method_no_deadlock_exp)
            mock_get.assert_called_once_with(1, True)

    def test_wrap_db_retry_get_interval(self):
        x = api.wrap_db_retry(max_retries=5, retry_on_deadlock=True, max_retry_interval=11)
        self.assertEqual(11, x.max_retry_interval)
        for i in (1, 2, 4):
            sleep_time, n = x._get_inc_interval(i, True)
            self.assertEqual(2 * i, n)
            self.assertTrue(2 * i > sleep_time)
            sleep_time, n = x._get_inc_interval(i, False)
            self.assertEqual(2 * i, n)
            self.assertEqual(2 * i, sleep_time)
        for i in (8, 16, 32):
            sleep_time, n = x._get_inc_interval(i, False)
            self.assertEqual(x.max_retry_interval, sleep_time)
            self.assertEqual(2 * i, n)
            sleep_time, n = x._get_inc_interval(i, True)
            self.assertTrue(x.max_retry_interval >= sleep_time)
            self.assertEqual(2 * i, n)