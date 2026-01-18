import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OpenIdClientLoginHandler(RequestHandler, OpenIdMixin):

    def initialize(self, test):
        self._OPENID_ENDPOINT = test.get_url('/openid/server/authenticate')

    @gen.coroutine
    def get(self):
        if self.get_argument('openid.mode', None):
            user = (yield self.get_authenticated_user(http_client=self.settings['http_client']))
            if user is None:
                raise Exception('user is None')
            self.finish(user)
            return
        res = self.authenticate_redirect()
        assert res is None