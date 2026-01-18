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
class DocXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    """XML-RPC and documentation request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.

    Handles all HTTP GET requests and interprets them as requests
    for documentation.
    """

    def _get_css(self, url):
        path_here = os.path.dirname(os.path.realpath(__file__))
        css_path = os.path.join(path_here, '..', 'pydoc_data', '_pydoc.css')
        with open(css_path, mode='rb') as fp:
            return fp.read()

    def do_GET(self):
        """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """
        if not self.is_rpc_path_valid():
            self.report_404()
            return
        if self.path.endswith('.css'):
            content_type = 'text/css'
            response = self._get_css(self.path)
        else:
            content_type = 'text/html'
            response = self.server.generate_html_documentation().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', '%s; charset=UTF-8' % content_type)
        self.send_header('Content-length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)