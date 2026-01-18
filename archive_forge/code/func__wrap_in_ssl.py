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
def _wrap_in_ssl(self, sock):
    """
        Given a single socket, wraps it in TLS.
        """
    return ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_SSLv23, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.ca_path, certfile=self.cert_path, keyfile=self.key_path, server_side=True)