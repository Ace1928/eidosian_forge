import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
class ValidatorTest(testtools.TestCase):

    def _test_validate_int(self, attr_val, attr_name='attr1', min_value=1, max_value=10):
        obj = FakeParsedArgs()
        setattr(obj, attr_name, attr_val)
        ret = validators.validate_int_range(obj, attr_name, min_value, max_value)
        self.assertIsNone(ret)

    def _test_validate_int_error(self, attr_val, expected_msg, attr_name='attr1', expected_exc=None, min_value=1, max_value=10):
        if expected_exc is None:
            expected_exc = exceptions.CommandError
        e = self.assertRaises(expected_exc, self._test_validate_int, attr_val, attr_name, min_value, max_value)
        self.assertEqual(expected_msg, str(e))

    def test_validate_int_min_max(self):
        self._test_validate_int(1)
        self._test_validate_int(10)
        self._test_validate_int('1')
        self._test_validate_int('10')
        self._test_validate_int('0x0a')
        self._test_validate_int_error(0, 'attr1 "0" should be an integer [1:10].')
        self._test_validate_int_error(11, 'attr1 "11" should be an integer [1:10].')
        self._test_validate_int_error('0x10', 'attr1 "0x10" should be an integer [1:10].')

    def test_validate_int_min_only(self):
        self._test_validate_int(1, max_value=None)
        self._test_validate_int(10, max_value=None)
        self._test_validate_int(11, max_value=None)
        self._test_validate_int_error(0, 'attr1 "0" should be an integer greater than or equal to 1.', max_value=None)

    def test_validate_int_max_only(self):
        self._test_validate_int(0, min_value=None)
        self._test_validate_int(1, min_value=None)
        self._test_validate_int(10, min_value=None)
        self._test_validate_int_error(11, 'attr1 "11" should be an integer smaller than or equal to 10.', min_value=None)

    def test_validate_int_no_limit(self):
        self._test_validate_int(0, min_value=None, max_value=None)
        self._test_validate_int(1, min_value=None, max_value=None)
        self._test_validate_int(10, min_value=None, max_value=None)
        self._test_validate_int(11, min_value=None, max_value=None)
        self._test_validate_int_error('abc', 'attr1 "abc" should be an integer.', min_value=None, max_value=None)

    def _test_validate_subnet(self, attr_val, attr_name='attr1'):
        obj = FakeParsedArgs()
        setattr(obj, attr_name, attr_val)
        ret = validators.validate_ip_subnet(obj, attr_name)
        self.assertIsNone(ret)

    def test_validate_ip_subnet(self):
        self._test_validate_subnet('192.168.2.0/24')
        self._test_validate_subnet('192.168.2.3/20')
        self._test_validate_subnet('192.168.2.1')
        e = self.assertRaises(exceptions.CommandError, self._test_validate_subnet, '192.168.2.256')
        self.assertEqual('attr1 "192.168.2.256" is not a valid CIDR.', str(e))