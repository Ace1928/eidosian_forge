import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
def _file_tail(self, relpath, tail_amount):
    t = self.get_readonly_transport()
    code, data = t._get(relpath, [], tail_amount)
    self.assertTrue(code in (200, 206), '_get returns: %d' % code)
    data.seek(-tail_amount, 2)
    return data.read(tail_amount)