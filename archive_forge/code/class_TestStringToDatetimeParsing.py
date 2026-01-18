from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
class TestStringToDatetimeParsing(unittest.TestCase):
    """ Test string to datetime parsing """

    def setUp(self):
        self._saved = locale.setlocale(locale.LC_ALL)
        try:
            locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
        except locale.Error:
            self.skipTest('Unsupported locale setting')

    def tearDown(self):
        locale.setlocale(locale.LC_ALL, self._saved)

    def test_nonus_locale(self):
        test_string = 'Thu, 15 May 2014 09:06:03 GMT'
        with self.assertRaises(ValueError):
            datetime.datetime.strptime(test_string, boto.utils.RFC1123)
        result = boto.utils.parse_ts(test_string)
        self.assertEqual(2014, result.year)
        self.assertEqual(5, result.month)
        self.assertEqual(15, result.day)
        self.assertEqual(9, result.hour)
        self.assertEqual(6, result.minute)