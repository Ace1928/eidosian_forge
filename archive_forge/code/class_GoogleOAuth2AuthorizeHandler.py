import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class GoogleOAuth2AuthorizeHandler(RequestHandler):

    def get(self):
        code = 'fake-authorization-code'
        self.redirect(url_concat(self.get_argument('redirect_uri'), dict(code=code)))