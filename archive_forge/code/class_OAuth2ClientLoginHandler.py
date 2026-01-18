import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OAuth2ClientLoginHandler(RequestHandler, OAuth2Mixin):

    def initialize(self, test):
        self._OAUTH_AUTHORIZE_URL = test.get_url('/oauth2/server/authorize')

    def get(self):
        res = self.authorize_redirect()
        assert res is None