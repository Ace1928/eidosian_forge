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
class TestingHTTPServerMixin:

    def __init__(self, test_case_server):
        self.test_case_server = test_case_server
        self._home_dir = test_case_server._home_dir