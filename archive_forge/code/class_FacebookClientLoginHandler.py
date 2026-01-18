import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class FacebookClientLoginHandler(RequestHandler, FacebookGraphMixin):

    def initialize(self, test):
        self._OAUTH_AUTHORIZE_URL = test.get_url('/facebook/server/authorize')
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url('/facebook/server/access_token')
        self._FACEBOOK_BASE_URL = test.get_url('/facebook/server')

    @gen.coroutine
    def get(self):
        if self.get_argument('code', None):
            user = (yield self.get_authenticated_user(redirect_uri=self.request.full_url(), client_id=self.settings['facebook_api_key'], client_secret=self.settings['facebook_secret'], code=self.get_argument('code')))
            self.write(user)
        else:
            self.authorize_redirect(redirect_uri=self.request.full_url(), client_id=self.settings['facebook_api_key'], extra_params={'scope': 'read_stream,offline_access'})