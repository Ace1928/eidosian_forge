import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@pytest.mark.skipif(not HAS_IPV6, reason='IPv6 not available')
class IPv6HTTPDummyServerTestCase(HTTPDummyServerTestCase):
    host = '::1'