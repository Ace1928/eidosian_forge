import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OAuth1ClientRequestParametersHandler(RequestHandler, OAuthMixin):

    def initialize(self, version):
        self._OAUTH_VERSION = version

    def _oauth_consumer_token(self):
        return dict(key='asdf', secret='qwer')

    def get(self):
        params = self._oauth_request_parameters('http://www.example.com/api/asdf', dict(key='uiop', secret='5678'), parameters=dict(foo='bar'))
        self.write(params)