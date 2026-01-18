from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
def _get_named_handler(handler_name):

    class Handler(RequestHandler):

        def get(self, *args, **kwargs):
            if self.application.settings.get('app_name') is not None:
                self.write(self.application.settings['app_name'] + ': ')
            self.finish(handler_name + ': ' + self.reverse_url(handler_name))
    return Handler