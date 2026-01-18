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
class TestHostIsIPV6(unittest.TestCase):

    def test_is_ipv6_no_brackets(self):
        hostname = 'bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_is_ipv6_with_brackets(self):
        hostname = '[bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be]'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_is_ipv6_with_brackets_and_port(self):
        hostname = '[bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be]:8080'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_is_ipv6_no_brackets_abbreviated(self):
        hostname = 'bf1d:cb48:4513::'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_is_ipv6_with_brackets_abbreviated(self):
        hostname = '[bf1d:cb48:4513::'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_is_ipv6_with_brackets_and_port_abbreviated(self):
        hostname = '[bf1d:cb48:4513::]:8080'
        result = boto.utils.host_is_ipv6(hostname)
        self.assertTrue(result)

    def test_empty_string(self):
        result = boto.utils.host_is_ipv6('')
        self.assertFalse(result)

    def test_not_of_string_type(self):
        hostnames = [None, 0, False, [], {}]
        for h in hostnames:
            result = boto.utils.host_is_ipv6(h)
            self.assertFalse(result)

    def test_ipv4_no_port(self):
        result = boto.utils.host_is_ipv6('192.168.1.1')
        self.assertFalse(result)

    def test_ipv4_with_port(self):
        result = boto.utils.host_is_ipv6('192.168.1.1:8080')
        self.assertFalse(result)

    def test_hostnames_are_not_ipv6_with_port(self):
        result = boto.utils.host_is_ipv6('example.org:8080')
        self.assertFalse(result)

    def test_hostnames_are_not_ipv6_without_port(self):
        result = boto.utils.host_is_ipv6('example.org')
        self.assertFalse(result)