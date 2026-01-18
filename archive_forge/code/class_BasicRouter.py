from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class BasicRouter(Router):

    def find_handler(self, request, **kwargs):

        class MessageDelegate(HTTPMessageDelegate):

            def __init__(self, connection):
                self.connection = connection

            def finish(self):
                self.connection.write_headers(ResponseStartLine('HTTP/1.1', 200, 'OK'), HTTPHeaders({'Content-Length': '2'}), b'OK')
                self.connection.finish()
        return MessageDelegate(request.connection)