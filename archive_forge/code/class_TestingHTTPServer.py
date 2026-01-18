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
class TestingHTTPServer(test_server.TestingTCPServer, TestingHTTPServerMixin):

    def __init__(self, server_address, request_handler_class, test_case_server):
        test_server.TestingTCPServer.__init__(self, server_address, request_handler_class)
        TestingHTTPServerMixin.__init__(self, test_case_server)