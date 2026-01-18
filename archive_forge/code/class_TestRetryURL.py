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
class TestRetryURL(unittest.TestCase):

    def setUp(self):
        self.urlopen_patch = mock.patch('boto.compat.urllib.request.urlopen')
        self.opener_patch = mock.patch('boto.compat.urllib.request.build_opener')
        self.urlopen = self.urlopen_patch.start()
        self.opener = self.opener_patch.start()

    def tearDown(self):
        self.urlopen_patch.stop()
        self.opener_patch.stop()

    def set_normal_response(self, response):
        fake_response = mock.Mock()
        fake_response.read.return_value = response
        self.urlopen.return_value = fake_response

    def set_no_proxy_allowed_response(self, response):
        fake_response = mock.Mock()
        fake_response.read.return_value = response
        self.opener.return_value.open.return_value = fake_response

    def test_retry_url_uses_proxy(self):
        self.set_normal_response('normal response')
        self.set_no_proxy_allowed_response('no proxy response')
        response = retry_url('http://10.10.10.10/foo', num_retries=1)
        self.assertEqual(response, 'no proxy response')

    def test_retry_url_using_bytes_and_string_response(self):
        test_value = 'normal response'
        fake_response = mock.Mock()
        fake_response.read.return_value = test_value
        self.opener.return_value.open.return_value = fake_response
        response = retry_url('http://10.10.10.10/foo', num_retries=1)
        self.assertEqual(response, test_value)
        fake_response.read.return_value = test_value.encode('utf-8')
        self.opener.return_value.open.return_value = fake_response
        response = retry_url('http://10.10.10.10/foo', num_retries=1)
        self.assertEqual(response, test_value)