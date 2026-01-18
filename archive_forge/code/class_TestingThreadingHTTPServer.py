import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
class TestingThreadingHTTPServer(test_server.TestingThreadingTCPServer, TestingHTTPServerMixin):
    """A threading HTTP test server for HTTP 1.1.

    Since tests can initiate several concurrent connections to the same http
    server, we need an independent connection for each of them. We achieve that
    by spawning a new thread for each connection.
    """

    def __init__(self, server_address, request_handler_class, test_case_server):
        test_server.TestingThreadingTCPServer.__init__(self, server_address, request_handler_class)
        TestingHTTPServerMixin.__init__(self, test_case_server)