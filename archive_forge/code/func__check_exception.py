import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
def _check_exception(self, exc_class, expected_msg, **kwargs):
    raise_exc_class = functools.partial(test_exceptions._raise, exc_class)
    e = self.assertRaises(exc_class, raise_exc_class, **kwargs)
    self.assertEqual(expected_msg, str(e))