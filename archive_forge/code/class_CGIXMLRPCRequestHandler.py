from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
class CGIXMLRPCRequestHandler(SimpleXMLRPCDispatcher):
    """Simple handler for XML-RPC data passed through CGI."""

    def __init__(self, allow_none=False, encoding=None, use_builtin_types=False):
        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding, use_builtin_types)

    def handle_xmlrpc(self, request_text):
        """Handle a single XML-RPC request"""
        response = self._marshaled_dispatch(request_text)
        print('Content-Type: text/xml')
        print('Content-Length: %d' % len(response))
        print()
        sys.stdout.flush()
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    def handle_get(self):
        """Handle a single HTTP GET request.

        Default implementation indicates an error because
        XML-RPC uses the POST method.
        """
        code = 400
        message, explain = BaseHTTPRequestHandler.responses[code]
        response = http.server.DEFAULT_ERROR_MESSAGE % {'code': code, 'message': message, 'explain': explain}
        response = response.encode('utf-8')
        print('Status: %d %s' % (code, message))
        print('Content-Type: %s' % http.server.DEFAULT_ERROR_CONTENT_TYPE)
        print('Content-Length: %d' % len(response))
        print()
        sys.stdout.flush()
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    def handle_request(self, request_text=None):
        """Handle a single XML-RPC request passed through a CGI post method.

        If no XML data is given then it is read from stdin. The resulting
        XML-RPC response is printed to stdout along with the correct HTTP
        headers.
        """
        if request_text is None and os.environ.get('REQUEST_METHOD', None) == 'GET':
            self.handle_get()
        else:
            try:
                length = int(os.environ.get('CONTENT_LENGTH', None))
            except (ValueError, TypeError):
                length = -1
            if request_text is None:
                request_text = sys.stdin.read(length)
            self.handle_xmlrpc(request_text)