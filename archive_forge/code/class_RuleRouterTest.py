from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class RuleRouterTest(AsyncHTTPTestCase):

    def get_app(self):
        app = Application()

        def request_callable(request):
            request.connection.write_headers(ResponseStartLine('HTTP/1.1', 200, 'OK'), HTTPHeaders({'Content-Length': '2'}))
            request.connection.write(b'OK')
            request.connection.finish()
        router = CustomRouter()
        router.add_routes({'/nested_handler': (app, _get_named_handler('nested_handler'))})
        app.add_handlers('.*', [(HostMatches('www.example.com'), [(PathMatches('/first_handler'), 'tornado.test.routing_test.SecondHandler', {}, 'second_handler')]), Rule(PathMatches('/.*handler'), router), Rule(PathMatches('/first_handler'), FirstHandler, name='first_handler'), Rule(PathMatches('/request_callable'), request_callable), ('/connection_delegate', ConnectionDelegate())])
        return app

    def test_rule_based_router(self):
        response = self.fetch('/first_handler')
        self.assertEqual(response.body, b'first_handler: /first_handler')
        response = self.fetch('/first_handler', headers={'Host': 'www.example.com'})
        self.assertEqual(response.body, b'second_handler: /first_handler')
        response = self.fetch('/nested_handler')
        self.assertEqual(response.body, b'nested_handler: /nested_handler')
        response = self.fetch('/nested_not_found_handler')
        self.assertEqual(response.code, 404)
        response = self.fetch('/connection_delegate')
        self.assertEqual(response.body, b'OK')
        response = self.fetch('/request_callable')
        self.assertEqual(response.body, b'OK')
        response = self.fetch('/404')
        self.assertEqual(response.code, 404)