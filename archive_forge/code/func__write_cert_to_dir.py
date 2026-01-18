import collections
import contextlib
import platform
import socket
import ssl
import sys
import threading
import pytest
import trustme
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import HAS_IPV6, run_tornado_app
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3.util import ssl_
from .tz_stub import stub_timezone_ctx
def _write_cert_to_dir(cert, tmpdir, file_prefix='server'):
    cert_path = str(tmpdir / ('%s.pem' % file_prefix))
    key_path = str(tmpdir / ('%s.key' % file_prefix))
    cert.private_key_pem.write_to_path(key_path)
    cert.cert_chain_pems[0].write_to_path(cert_path)
    certs = {'keyfile': key_path, 'certfile': cert_path}
    return certs