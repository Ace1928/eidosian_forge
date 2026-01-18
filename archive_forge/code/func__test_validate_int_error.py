import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def _test_validate_int_error(self, attr_val, expected_msg, attr_name='attr1', expected_exc=None, min_value=1, max_value=10):
    if expected_exc is None:
        expected_exc = exceptions.CommandError
    e = self.assertRaises(expected_exc, self._test_validate_int, attr_val, attr_name, min_value, max_value)
    self.assertEqual(expected_msg, str(e))