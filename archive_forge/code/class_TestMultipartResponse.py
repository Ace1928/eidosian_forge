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
class TestMultipartResponse(SocketDummyServerTestCase):

    def test_multipart_assert_header_parsing_no_defects(self):

        def socket_handler(listener):
            for _ in range(2):
                sock = listener.accept()[0]
                while not sock.recv(65536).endswith(b'\r\n\r\n'):
                    pass
                sock.sendall(b'HTTP/1.1 404 Not Found\r\nServer: example.com\r\nContent-Type: multipart/mixed; boundary=36eeb8c4e26d842a\r\nContent-Length: 73\r\n\r\n--36eeb8c4e26d842a\r\nContent-Type: text/plain\r\n\r\n1\r\n--36eeb8c4e26d842a--\r\n')
                sock.close()
        self._start_server(socket_handler)
        from urllib3.connectionpool import log
        with mock.patch.object(log, 'warning') as log_warning:
            with HTTPConnectionPool(self.host, self.port, timeout=3) as pool:
                resp = pool.urlopen('GET', '/')
                assert resp.status == 404
                assert resp.headers['content-type'] == 'multipart/mixed; boundary=36eeb8c4e26d842a'
                assert len(resp.data) == 73
                log_warning.assert_not_called()