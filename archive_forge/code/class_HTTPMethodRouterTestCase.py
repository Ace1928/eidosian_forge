from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class HTTPMethodRouterTestCase(AsyncHTTPTestCase):

    def get_app(self):
        return HTTPMethodRouter(Application())

    def test_http_method_router(self):
        response = self.fetch('/post_resource', method='POST', body='data')
        self.assertEqual(response.code, 200)
        response = self.fetch('/get_resource')
        self.assertEqual(response.code, 404)
        response = self.fetch('/post_resource')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'data')