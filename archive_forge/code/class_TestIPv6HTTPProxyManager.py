import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
import pytest
import trustme
from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.ssl_ import create_urllib3_context
from .. import TARPIT_HOST, requires_network
@pytest.mark.skipif(not HAS_IPV6, reason='Only runs on IPv6 systems')
class TestIPv6HTTPProxyManager(IPv6HTTPDummyProxyTestCase):

    @classmethod
    def setup_class(cls):
        HTTPDummyProxyTestCase.setup_class()
        cls.http_url = 'http://%s:%d' % (cls.http_host, cls.http_port)
        cls.http_url_alt = 'http://%s:%d' % (cls.http_host_alt, cls.http_port)
        cls.https_url = 'https://%s:%d' % (cls.https_host, cls.https_port)
        cls.https_url_alt = 'https://%s:%d' % (cls.https_host_alt, cls.https_port)
        cls.proxy_url = 'http://[%s]:%d' % (cls.proxy_host, cls.proxy_port)

    def test_basic_ipv6_proxy(self):
        with proxy_from_url(self.proxy_url, ca_certs=DEFAULT_CA) as http:
            r = http.request('GET', '%s/' % self.http_url)
            assert r.status == 200
            r = http.request('GET', '%s/' % self.https_url)
            assert r.status == 200