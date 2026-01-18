import io
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
from test import LONG_TIMEOUT, SHORT_TIMEOUT, onlyPy2
from threading import Event
import mock
import pytest
import six
from dummyserver.server import HAS_IPV6_AND_DNS, NoIPv6Warning
from dummyserver.testcase import HTTPDummyServerTestCase, SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, encode_multipart_formdata
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six import b, u
from urllib3.packages.six.moves.urllib.parse import urlencode
from urllib3.util import SKIP_HEADER, SKIPPABLE_HEADERS
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout
from .. import INVALID_SOURCE_ADDRESSES, TARPIT_HOST, VALID_SOURCE_ADDRESSES
from ..port_helpers import find_unused_port
class TestRetry(HTTPDummyServerTestCase):

    def test_max_retry(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            with pytest.raises(MaxRetryError):
                pool.request('GET', '/redirect', fields={'target': '/'}, retries=0)

    def test_disabled_retry(self):
        """Disabled retries should disable redirect handling."""
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/redirect', fields={'target': '/'}, retries=False)
            assert r.status == 303
            r = pool.request('GET', '/redirect', fields={'target': '/'}, retries=Retry(redirect=False))
            assert r.status == 303
        with HTTPConnectionPool('thishostdoesnotexist.invalid', self.port, timeout=0.001) as pool:
            with pytest.raises(NewConnectionError):
                pool.request('GET', '/test', retries=False)

    def test_read_retries(self):
        """Should retry for status codes in the whitelist"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            retry = Retry(read=1, status_forcelist=[418])
            resp = pool.request('GET', '/successful_retry', headers={'test-name': 'test_read_retries'}, retries=retry)
            assert resp.status == 200

    def test_read_total_retries(self):
        """HTTP response w/ status code in the whitelist should be retried"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'test-name': 'test_read_total_retries'}
            retry = Retry(total=1, status_forcelist=[418])
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=retry)
            assert resp.status == 200

    def test_retries_wrong_whitelist(self):
        """HTTP response w/ status code not in whitelist shouldn't be retried"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            retry = Retry(total=1, status_forcelist=[202])
            resp = pool.request('GET', '/successful_retry', headers={'test-name': 'test_wrong_whitelist'}, retries=retry)
            assert resp.status == 418

    def test_default_method_whitelist_retried(self):
        """urllib3 should retry methods in the default method whitelist"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            retry = Retry(total=1, status_forcelist=[418])
            resp = pool.request('OPTIONS', '/successful_retry', headers={'test-name': 'test_default_whitelist'}, retries=retry)
            assert resp.status == 200

    def test_retries_wrong_method_list(self):
        """Method not in our whitelist should not be retried, even if code matches"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'test-name': 'test_wrong_method_whitelist'}
            retry = Retry(total=1, status_forcelist=[418], method_whitelist=['POST'])
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=retry)
            assert resp.status == 418

    def test_read_retries_unsuccessful(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'test-name': 'test_read_retries_unsuccessful'}
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=1)
            assert resp.status == 418

    def test_retry_reuse_safe(self):
        """It should be possible to reuse a Retry object across requests"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'test-name': 'test_retry_safe'}
            retry = Retry(total=1, status_forcelist=[418])
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=retry)
            assert resp.status == 200
        with HTTPConnectionPool(self.host, self.port) as pool:
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=retry)
            assert resp.status == 200

    def test_retry_return_in_response(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'test-name': 'test_retry_return_in_response'}
            retry = Retry(total=2, status_forcelist=[418])
            resp = pool.request('GET', '/successful_retry', headers=headers, retries=retry)
            assert resp.status == 200
            assert resp.retries.total == 1
            assert resp.retries.history == (RequestHistory('GET', '/successful_retry', None, 418, None),)

    def test_retry_redirect_history(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            resp = pool.request('GET', '/redirect', fields={'target': '/'})
            assert resp.status == 200
            assert resp.retries.history == (RequestHistory('GET', '/redirect?target=%2F', None, 303, '/'),)

    def test_multi_redirect_history(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/multi_redirect', fields={'redirect_codes': '303,302,200'}, redirect=False)
            assert r.status == 303
            assert r.retries.history == tuple()
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/multi_redirect', retries=10, fields={'redirect_codes': '303,302,301,307,302,200'})
            assert r.status == 200
            assert r.data == b'Done redirecting'
            expected = [(303, '/multi_redirect?redirect_codes=302,301,307,302,200'), (302, '/multi_redirect?redirect_codes=301,307,302,200'), (301, '/multi_redirect?redirect_codes=307,302,200'), (307, '/multi_redirect?redirect_codes=302,200'), (302, '/multi_redirect?redirect_codes=200')]
            actual = [(history.status, history.redirect_location) for history in r.retries.history]
            assert actual == expected