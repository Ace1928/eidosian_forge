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
class TestFileBodiesOnRetryOrRedirect(HTTPDummyServerTestCase):

    def test_retries_put_filehandle(self):
        """HTTP PUT retry with a file-like object should not timeout"""
        with HTTPConnectionPool(self.host, self.port, timeout=0.1) as pool:
            retry = Retry(total=3, status_forcelist=[418])
            content_length = 65535
            data = b'A' * content_length
            uploaded_file = io.BytesIO(data)
            headers = {'test-name': 'test_retries_put_filehandle', 'Content-Length': str(content_length)}
            resp = pool.urlopen('PUT', '/successful_retry', headers=headers, retries=retry, body=uploaded_file, assert_same_host=False, redirect=False)
            assert resp.status == 200

    def test_redirect_put_file(self):
        """PUT with file object should work with a redirection response"""
        with HTTPConnectionPool(self.host, self.port, timeout=0.1) as pool:
            retry = Retry(total=3, status_forcelist=[418])
            content_length = 65535
            data = b'A' * content_length
            uploaded_file = io.BytesIO(data)
            headers = {'test-name': 'test_redirect_put_file', 'Content-Length': str(content_length)}
            url = '/redirect?target=/echo&status=307'
            resp = pool.urlopen('PUT', url, headers=headers, retries=retry, body=uploaded_file, assert_same_host=False, redirect=True)
            assert resp.status == 200
            assert resp.data == data

    def test_redirect_with_failed_tell(self):
        """Abort request if failed to get a position from tell()"""

        class BadTellObject(io.BytesIO):

            def tell(self):
                raise IOError
        body = BadTellObject(b'the data')
        url = '/redirect?target=/successful_retry'
        headers = {'Content-Length': '8'}
        with HTTPConnectionPool(self.host, self.port, timeout=0.1) as pool:
            with pytest.raises(UnrewindableBodyError) as e:
                pool.urlopen('PUT', url, headers=headers, body=body)
            assert 'Unable to record file position for' in str(e.value)