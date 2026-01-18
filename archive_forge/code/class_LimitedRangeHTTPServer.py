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
class LimitedRangeHTTPServer(http_server.HttpServer):
    """An HttpServer erroring out on requests with too much range specifiers"""

    def __init__(self, request_handler=LimitedRangeRequestHandler, protocol_version=None, range_limit=None):
        http_server.HttpServer.__init__(self, request_handler, protocol_version=protocol_version)
        self.range_limit = range_limit