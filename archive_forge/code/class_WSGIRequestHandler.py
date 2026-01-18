from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import urllib.parse
from wsgiref.handlers import SimpleHandler
from platform import python_implementation
class WSGIRequestHandler(BaseHTTPRequestHandler):
    server_version = 'WSGIServer/' + __version__

    def get_environ(self):
        env = self.server.base_environ.copy()
        env['SERVER_PROTOCOL'] = self.request_version
        env['SERVER_SOFTWARE'] = self.server_version
        env['REQUEST_METHOD'] = self.command
        if '?' in self.path:
            path, query = self.path.split('?', 1)
        else:
            path, query = (self.path, '')
        env['PATH_INFO'] = urllib.parse.unquote(path, 'iso-8859-1')
        env['QUERY_STRING'] = query
        host = self.address_string()
        if host != self.client_address[0]:
            env['REMOTE_HOST'] = host
        env['REMOTE_ADDR'] = self.client_address[0]
        if self.headers.get('content-type') is None:
            env['CONTENT_TYPE'] = self.headers.get_content_type()
        else:
            env['CONTENT_TYPE'] = self.headers['content-type']
        length = self.headers.get('content-length')
        if length:
            env['CONTENT_LENGTH'] = length
        for k, v in self.headers.items():
            k = k.replace('-', '_').upper()
            v = v.strip()
            if k in env:
                continue
            if 'HTTP_' + k in env:
                env['HTTP_' + k] += ',' + v
            else:
                env['HTTP_' + k] = v
        return env

    def get_stderr(self):
        return sys.stderr

    def handle(self):
        """Handle a single HTTP request"""
        self.raw_requestline = self.rfile.readline(65537)
        if len(self.raw_requestline) > 65536:
            self.requestline = ''
            self.request_version = ''
            self.command = ''
            self.send_error(414)
            return
        if not self.parse_request():
            return
        handler = ServerHandler(self.rfile, self.wfile, self.get_stderr(), self.get_environ(), multithread=False)
        handler.request_handler = self
        handler.run(self.server.get_app())