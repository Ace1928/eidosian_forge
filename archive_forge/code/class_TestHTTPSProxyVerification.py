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
class TestHTTPSProxyVerification:

    @onlyPy3
    def test_https_proxy_hostname_verification(self, no_localhost_san_server):
        bad_server = no_localhost_san_server
        bad_proxy_url = 'https://%s:%s' % (bad_server.host, bad_server.port)
        test_url = 'testing.com'
        with proxy_from_url(bad_proxy_url, ca_certs=bad_server.ca_certs) as https:
            with pytest.raises(MaxRetryError) as e:
                https.request('GET', 'http://%s/' % test_url)
            assert isinstance(e.value.reason, SSLError)
            assert "hostname 'localhost' doesn't match" in str(e.value.reason)
            with pytest.raises(MaxRetryError) as e:
                https.request('GET', 'https://%s/' % test_url)
            assert isinstance(e.value.reason, SSLError)
            assert "hostname 'localhost' doesn't match" in str(e.value.reason) or 'Hostname mismatch' in str(e.value.reason)

    @onlyPy3
    def test_https_proxy_ipv4_san(self, ipv4_san_proxy):
        proxy, server = ipv4_san_proxy
        proxy_url = 'https://%s:%s' % (proxy.host, proxy.port)
        destination_url = 'https://%s:%s' % (server.host, server.port)
        with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
            r = https.request('GET', destination_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy_ipv6_san(self, ipv6_san_proxy):
        proxy, server = ipv6_san_proxy
        proxy_url = 'https://[%s]:%s' % (proxy.host, proxy.port)
        destination_url = 'https://%s:%s' % (server.host, server.port)
        with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
            r = https.request('GET', destination_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy_common_name_warning(self, no_san_proxy):
        proxy, server = no_san_proxy
        proxy_url = 'https://%s:%s' % (proxy.host, proxy.port)
        destination_url = 'https://%s:%s' % (server.host, server.port)
        with warnings.catch_warnings(record=True) as w:
            with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
                r = https.request('GET', destination_url)
                assert r.status == 200
        assert len(w) == 1
        assert w[0].category == SubjectAltNameWarning