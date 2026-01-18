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
class TestParseHost(unittest.TestCase):

    def test_parses_ipv6_hosts_no_brackets(self):
        host = 'bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, host)

    def test_parses_ipv6_hosts_with_brackets_stripping_them(self):
        host = '[bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be]'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, 'bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be')

    def test_parses_ipv6_hosts_with_brackets_and_port(self):
        host = '[bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be]:8080'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, 'bf1d:cb48:4513:d1f1:efdd:b290:9ff9:64be')

    def test_parses_ipv4_hosts(self):
        host = '10.0.1.1'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, host)

    def test_parses_ipv4_hosts_with_port(self):
        host = '192.168.168.200:8080'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, '192.168.168.200')

    def test_parses_hostnames_with_port(self):
        host = 'example.org:8080'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, 'example.org')

    def test_parses_hostnames_without_port(self):
        host = 'example.org'
        result = boto.utils.parse_host(host)
        self.assertEquals(result, host)