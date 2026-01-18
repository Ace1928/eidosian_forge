from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class CustomRouterTestCase(AsyncHTTPTestCase):

    def get_app(self):
        router = CustomRouter()

        class CustomApplication(Application):

            def reverse_url(self, name, *args):
                return router.reverse_url(name, *args)
        app1 = CustomApplication(app_name='app1')
        app2 = CustomApplication(app_name='app2')
        router.add_routes({'/first_handler': (app1, FirstHandler), '/second_handler': (app2, SecondHandler), '/first_handler_second_app': (app2, FirstHandler)})
        return router

    def test_custom_router(self):
        response = self.fetch('/first_handler')
        self.assertEqual(response.body, b'app1: first_handler: /first_handler')
        response = self.fetch('/second_handler')
        self.assertEqual(response.body, b'app2: second_handler: /second_handler')
        response = self.fetch('/first_handler_second_app')
        self.assertEqual(response.body, b'app2: first_handler: /first_handler')