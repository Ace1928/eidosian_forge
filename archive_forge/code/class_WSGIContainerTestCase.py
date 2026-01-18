from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class WSGIContainerTestCase(AsyncHTTPTestCase):

    def get_app(self):
        wsgi_app = WSGIContainer(self.wsgi_app)

        class Handler(RequestHandler):

            def get(self, *args, **kwargs):
                self.finish(self.reverse_url('tornado'))
        return RuleRouter([(PathMatches('/tornado.*'), Application([('/tornado/test', Handler, {}, 'tornado')])), (PathMatches('/wsgi'), wsgi_app)])

    def wsgi_app(self, environ, start_response):
        start_response('200 OK', [])
        return [b'WSGI']

    def test_wsgi_container(self):
        response = self.fetch('/tornado/test')
        self.assertEqual(response.body, b'/tornado/test')
        response = self.fetch('/wsgi')
        self.assertEqual(response.body, b'WSGI')

    def test_delegate_not_found(self):
        response = self.fetch('/404')
        self.assertEqual(response.code, 404)