import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
class WSGIHandlerMixin:
    """
    WSGI mix-in for HTTPRequestHandler

    This class is a mix-in to provide WSGI functionality to any
    HTTPRequestHandler derivative (as provided in Python's BaseHTTPServer).
    This assumes a ``wsgi_application`` handler on ``self.server``.
    """
    lookup_addresses = True

    def log_request(self, *args, **kwargs):
        """ disable success request logging

        Logging transactions should not be part of a WSGI server,
        if you want logging; look at paste.translogger
        """
        pass

    def log_message(self, *args, **kwargs):
        """ disable error message logging

        Logging transactions should not be part of a WSGI server,
        if you want logging; look at paste.translogger
        """
        pass

    def version_string(self):
        """ behavior that BaseHTTPServer should have had """
        if not self.sys_version:
            return self.server_version
        else:
            return self.server_version + ' ' + self.sys_version

    def wsgi_write_chunk(self, chunk):
        """
        Write a chunk of the output stream; send headers if they
        have not already been sent.
        """
        if not self.wsgi_headers_sent and (not self.wsgi_curr_headers):
            raise RuntimeError('Content returned before start_response called')
        if not self.wsgi_headers_sent:
            self.wsgi_headers_sent = True
            status, headers = self.wsgi_curr_headers
            code, message = status.split(' ', 1)
            self.send_response(int(code), message)
            send_close = True
            for k, v in headers:
                lk = k.lower()
                if 'content-length' == lk:
                    send_close = False
                if 'connection' == lk:
                    if 'close' == v.lower():
                        self.close_connection = 1
                        send_close = False
                self.send_header(k, v)
            if send_close:
                self.close_connection = 1
                self.send_header('Connection', 'close')
            self.end_headers()
        self.wfile.write(chunk)

    def wsgi_start_response(self, status, response_headers, exc_info=None):
        if exc_info:
            try:
                if self.wsgi_headers_sent:
                    raise exc_info
                else:
                    pass
            finally:
                exc_info = None
        elif self.wsgi_curr_headers:
            assert 0, 'Attempt to set headers a second time w/o an exc_info'
        self.wsgi_curr_headers = (status, response_headers)
        return self.wsgi_write_chunk

    def wsgi_setup(self, environ=None):
        """
        Setup the member variables used by this WSGI mixin, including
        the ``environ`` and status member variables.

        After the basic environment is created; the optional ``environ``
        argument can be used to override any settings.
        """
        dummy_url = 'http://dummy%s' % (self.path,)
        scheme, netloc, path, query, fragment = urlsplit(dummy_url)
        path = unquote(path)
        endslash = path.endswith('/')
        path = posixpath.normpath(path)
        if endslash and path != '/':
            path += '/'
        server_name, server_port = self.server.server_address[:2]
        rfile = self.rfile
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            content_length = 0
        if '100-continue' == self.headers.get('Expect', '').lower():
            rfile = LimitedLengthFile(ContinueHook(rfile, self.wfile.write), content_length)
        elif not hasattr(self.connection, 'get_context'):
            rfile = LimitedLengthFile(rfile, content_length)
        remote_address = self.client_address[0]
        self.wsgi_environ = {'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http', 'wsgi.input': rfile, 'wsgi.errors': sys.stderr, 'wsgi.multithread': True, 'wsgi.multiprocess': False, 'wsgi.run_once': False, 'REQUEST_METHOD': self.command, 'SCRIPT_NAME': '', 'PATH_INFO': path, 'QUERY_STRING': query, 'CONTENT_TYPE': self.headers.get('Content-Type', ''), 'CONTENT_LENGTH': self.headers.get('Content-Length', '0'), 'SERVER_NAME': server_name, 'SERVER_PORT': str(server_port), 'SERVER_PROTOCOL': self.request_version, 'REMOTE_ADDR': remote_address}
        if scheme:
            self.wsgi_environ['paste.httpserver.proxy.scheme'] = scheme
        if netloc:
            self.wsgi_environ['paste.httpserver.proxy.host'] = netloc
        if self.lookup_addresses:
            if remote_address.startswith('192.168.') or remote_address.startswith('10.') or remote_address.startswith('172.16.'):
                pass
            else:
                address_string = None
                if address_string:
                    self.wsgi_environ['REMOTE_HOST'] = address_string
        if hasattr(self.server, 'thread_pool'):
            self.server.thread_pool.worker_tracker[_thread.get_ident()][1] = self.wsgi_environ
            self.wsgi_environ['paste.httpserver.thread_pool'] = self.server.thread_pool
        for k, v in self.headers.items():
            key = 'HTTP_' + k.replace('-', '_').upper()
            if key in ('HTTP_CONTENT_TYPE', 'HTTP_CONTENT_LENGTH'):
                continue
            self.wsgi_environ[key] = ','.join(_get_headers(self.headers, k))
        if hasattr(self.connection, 'get_context'):
            self.wsgi_environ['wsgi.url_scheme'] = 'https'
        if environ:
            assert isinstance(environ, dict)
            self.wsgi_environ.update(environ)
            if 'on' == environ.get('HTTPS'):
                self.wsgi_environ['wsgi.url_scheme'] = 'https'
        self.wsgi_curr_headers = None
        self.wsgi_headers_sent = False

    def wsgi_connection_drop(self, exce, environ=None):
        """
        Override this if you're interested in socket exceptions, such
        as when the user clicks 'Cancel' during a file download.
        """
        pass

    def wsgi_execute(self, environ=None):
        """
        Invoke the server's ``wsgi_application``.
        """
        self.wsgi_setup(environ)
        try:
            result = self.server.wsgi_application(self.wsgi_environ, self.wsgi_start_response)
            try:
                for chunk in result:
                    self.wsgi_write_chunk(chunk)
                if not self.wsgi_headers_sent:
                    self.wsgi_write_chunk(b'')
            finally:
                if hasattr(result, 'close'):
                    result.close()
                result = None
        except socket.error as exce:
            self.wsgi_connection_drop(exce, environ)
            return
        except:
            if not self.wsgi_headers_sent:
                error_msg = 'Internal Server Error\n'
                self.wsgi_curr_headers = ('500 Internal Server Error', [('Content-type', 'text/plain'), ('Content-length', str(len(error_msg)))])
                self.wsgi_write_chunk(b'Internal Server Error\n')
            raise