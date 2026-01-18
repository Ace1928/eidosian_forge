from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class PostResource(RequestHandler):

    def post(self, path):
        resources[path] = self.request.body