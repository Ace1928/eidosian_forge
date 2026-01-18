import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
class DocHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        """Process a request from an HTML browser.

            The URL received is in self.path.
            Get an HTML page from self.urlhandler and send it.
            """
        if self.path.endswith('.css'):
            content_type = 'text/css'
        else:
            content_type = 'text/html'
        self.send_response(200)
        self.send_header('Content-Type', '%s; charset=UTF-8' % content_type)
        self.end_headers()
        self.wfile.write(self.urlhandler(self.path, content_type).encode('utf-8'))

    def log_message(self, *args):
        pass