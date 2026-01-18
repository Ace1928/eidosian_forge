import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OAuth1ClientLoginHandler(RequestHandler, OAuthMixin):

    def initialize(self, test, version):
        self._OAUTH_VERSION = version
        self._OAUTH_REQUEST_TOKEN_URL = test.get_url('/oauth1/server/request_token')
        self._OAUTH_AUTHORIZE_URL = test.get_url('/oauth1/server/authorize')
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url('/oauth1/server/access_token')

    def _oauth_consumer_token(self):
        return dict(key='asdf', secret='qwer')

    @gen.coroutine
    def get(self):
        if self.get_argument('oauth_token', None):
            user = (yield self.get_authenticated_user(http_client=self.settings['http_client']))
            if user is None:
                raise Exception('user is None')
            self.finish(user)
            return
        yield self.authorize_redirect(http_client=self.settings['http_client'])

    @gen.coroutine
    def _oauth_get_user_future(self, access_token):
        if self.get_argument('fail_in_get_user', None):
            raise Exception('failing in get_user')
        if access_token != dict(key='uiop', secret='5678'):
            raise Exception('incorrect access token %r' % access_token)
        return dict(email='foo@example.com')