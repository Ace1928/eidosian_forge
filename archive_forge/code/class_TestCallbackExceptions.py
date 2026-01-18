import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
class TestCallbackExceptions(test_exceptions.TestExceptions):

    def _check_exception(self, exc_class, expected_msg, **kwargs):
        raise_exc_class = functools.partial(test_exceptions._raise, exc_class)
        e = self.assertRaises(exc_class, raise_exc_class, **kwargs)
        self.assertEqual(expected_msg, str(e))

    def test_invalid(self):
        self._check_exception(ex.Invalid, "The value 'foo' for bar is not valid.", value='foo', element='bar')

    def test_callback_failure(self):
        self._check_exception(ex.CallbackFailure, 'one', errors='one')

    def test_callback_failure_with_list(self):
        self._check_exception(ex.CallbackFailure, '1,2,3', errors=[1, 2, 3])

    def test_notification_error(self):
        """Test that correct message is created for this error class."""
        error = ex.NotificationError('abc', 'boom')
        self.assertEqual('Callback abc failed with "boom"', str(error))

    def test_inner_exceptions(self):
        key_err = KeyError()
        n_key_err = ex.NotificationError('cb1', key_err)
        err = ex.CallbackFailure([key_err, n_key_err])
        self.assertEqual([key_err, n_key_err.error], err.inner_exceptions)