import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
class Controller(helper.Controller):
    """Controller for serving WSGI apps."""

    def hello(req, resp):
        """Render Hello world."""
        return 'Hello, world!'

    def pov(req, resp):
        """Render ``pov`` value."""
        return pov

    def stream(req, resp):
        """Render streaming response."""
        if 'set_cl' in req.environ['QUERY_STRING']:
            resp.headers['Content-Length'] = str(10)

        def content():
            for x in range(10):
                yield str(x)
        return content()

    def upload(req, resp):
        """Process file upload and render thank."""
        if not req.environ['REQUEST_METHOD'] == 'POST':
            raise AssertionError("'POST' != request.method %r" % req.environ['REQUEST_METHOD'])
        return "thanks for '%s'" % req.environ['wsgi.input'].read()

    def custom_204(req, resp):
        """Render response with status 204."""
        resp.status = '204'
        return 'Code = 204'

    def custom_304(req, resp):
        """Render response with status 304."""
        resp.status = '304'
        return 'Code = 304'

    def err_before_read(req, resp):
        """Render response with status 500."""
        resp.status = '500 Internal Server Error'
        return 'ok'

    def one_megabyte_of_a(req, resp):
        """Render 1MB response."""
        return ['a' * 1024] * 1024

    def wrong_cl_buffered(req, resp):
        """Render buffered response with invalid length value."""
        resp.headers['Content-Length'] = '5'
        return 'I have too many bytes'

    def wrong_cl_unbuffered(req, resp):
        """Render unbuffered response with invalid length value."""
        resp.headers['Content-Length'] = '5'
        return ['I too', ' have too many bytes']

    def _munge(string):
        """Encode PATH_INFO correctly depending on Python version.

        WSGI 1.0 is a mess around unicode. Create endpoints
        that match the PATH_INFO that it produces.
        """
        return string.encode('utf-8').decode('latin-1')
    handlers = {'/hello': hello, '/pov': pov, '/page1': pov, '/page2': pov, '/page3': pov, '/stream': stream, '/upload': upload, '/custom/204': custom_204, '/custom/304': custom_304, '/err_before_read': err_before_read, '/one_megabyte_of_a': one_megabyte_of_a, '/wrong_cl_buffered': wrong_cl_buffered, '/wrong_cl_unbuffered': wrong_cl_unbuffered}