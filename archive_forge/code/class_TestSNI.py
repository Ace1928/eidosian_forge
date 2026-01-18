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
class TestSNI(SocketDummyServerTestCase):

    def test_hostname_in_first_request_packet(self):
        if not util.HAS_SNI:
            pytest.skip('SNI-support not available')
        done_receiving = Event()
        self.buf = b''

        def socket_handler(listener):
            sock = listener.accept()[0]
            self.buf = sock.recv(65536)
            done_receiving.set()
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port) as pool:
            try:
                pool.request('GET', '/', retries=0)
            except MaxRetryError:
                pass
            successful = done_receiving.wait(LONG_TIMEOUT)
            assert successful, 'Timed out waiting for connection accept'
            assert self.host.encode('ascii') in self.buf, 'missing hostname in SSL handshake'