from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
@pytest.mark.skipif(issubclass(httplib.HTTPMessage, MimeToolMessage), reason='Header parsing errors not available')
class TestBrokenHeaders(SocketDummyServerTestCase):

    def _test_broken_header_parsing(self, headers, unparsed_data_check=None):
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-type: text/plain\r\n' + b'\r\n'.join(headers) + b'\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with LogRecorder() as logs:
                pool.request('GET', '/')
            for record in logs:
                if 'Failed to parse headers' in record.msg and pool._absolute_url('/') == record.args[0]:
                    if unparsed_data_check is None or unparsed_data_check in record.getMessage():
                        return
            self.fail('Missing log about unparsed headers')

    def test_header_without_name(self):
        self._test_broken_header_parsing([b': Value', b'Another: Header'])

    def test_header_without_name_or_value(self):
        self._test_broken_header_parsing([b':', b'Another: Header'])

    def test_header_without_colon_or_value(self):
        self._test_broken_header_parsing([b'Broken Header', b'Another: Header'], 'Broken Header')