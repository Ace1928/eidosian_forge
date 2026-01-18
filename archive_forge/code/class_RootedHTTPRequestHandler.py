from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import filter, str
from future import utils
import os
import sys
import ssl
import pprint
import socket
from future.backports.urllib import parse as urllib_parse
from future.backports.http.server import (HTTPServer as _HTTPServer,
from future.backports.test import support
class RootedHTTPRequestHandler(SimpleHTTPRequestHandler):
    server_version = 'TestHTTPS/1.0'
    root = here
    timeout = 5

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.

        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)

        """
        path = urllib.parse.urlparse(path)[2]
        path = os.path.normpath(urllib.parse.unquote(path))
        words = path.split('/')
        words = filter(None, words)
        path = self.root
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            path = os.path.join(path, word)
        return path

    def log_message(self, format, *args):
        if support.verbose:
            sys.stdout.write(' server (%s:%d %s):\n   [%s] %s\n' % (self.server.server_address, self.server.server_port, self.request.cipher(), self.log_date_time_string(), format % args))