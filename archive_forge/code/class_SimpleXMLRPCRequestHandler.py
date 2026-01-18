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
class SimpleXMLRPCRequestHandler(BaseHTTPRequestHandler):
    """Simple XML-RPC request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.
    """
    rpc_paths = ('/', '/RPC2', '/pydoc.css')
    encode_threshold = 1400
    wbufsize = -1
    disable_nagle_algorithm = True
    aepattern = re.compile('\n                            \\s* ([^\\s;]+) \\s*            #content-coding\n                            (;\\s* q \\s*=\\s* ([0-9\\.]+))? #q\n                            ', re.VERBOSE | re.IGNORECASE)

    def accept_encodings(self):
        r = {}
        ae = self.headers.get('Accept-Encoding', '')
        for e in ae.split(','):
            match = self.aepattern.match(e)
            if match:
                v = match.group(3)
                v = float(v) if v else 1.0
                r[match.group(1)] = v
        return r

    def is_rpc_path_valid(self):
        if self.rpc_paths:
            return self.path in self.rpc_paths
        else:
            return True

    def do_POST(self):
        """Handles the HTTP POST request.

        Attempts to interpret all HTTP POST requests as XML-RPC calls,
        which are forwarded to the server's _dispatch method for handling.
        """
        if not self.is_rpc_path_valid():
            self.report_404()
            return
        try:
            max_chunk_size = 10 * 1024 * 1024
            size_remaining = int(self.headers['content-length'])
            L = []
            while size_remaining:
                chunk_size = min(size_remaining, max_chunk_size)
                chunk = self.rfile.read(chunk_size)
                if not chunk:
                    break
                L.append(chunk)
                size_remaining -= len(L[-1])
            data = b''.join(L)
            data = self.decode_request_content(data)
            if data is None:
                return
            response = self.server._marshaled_dispatch(data, getattr(self, '_dispatch', None), self.path)
        except Exception as e:
            self.send_response(500)
            if hasattr(self.server, '_send_traceback_header') and self.server._send_traceback_header:
                self.send_header('X-exception', str(e))
                trace = traceback.format_exc()
                trace = str(trace.encode('ASCII', 'backslashreplace'), 'ASCII')
                self.send_header('X-traceback', trace)
            self.send_header('Content-length', '0')
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/xml')
            if self.encode_threshold is not None:
                if len(response) > self.encode_threshold:
                    q = self.accept_encodings().get('gzip', 0)
                    if q:
                        try:
                            response = gzip_encode(response)
                            self.send_header('Content-Encoding', 'gzip')
                        except NotImplementedError:
                            pass
            self.send_header('Content-length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    def decode_request_content(self, data):
        encoding = self.headers.get('content-encoding', 'identity').lower()
        if encoding == 'identity':
            return data
        if encoding == 'gzip':
            try:
                return gzip_decode(data)
            except NotImplementedError:
                self.send_response(501, 'encoding %r not supported' % encoding)
            except ValueError:
                self.send_response(400, 'error decoding gzip content')
        else:
            self.send_response(501, 'encoding %r not supported' % encoding)
        self.send_header('Content-length', '0')
        self.end_headers()

    def report_404(self):
        self.send_response(404)
        response = b'No such page'
        self.send_header('Content-type', 'text/plain')
        self.send_header('Content-length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""
        if self.server.logRequests:
            BaseHTTPRequestHandler.log_request(self, code, size)