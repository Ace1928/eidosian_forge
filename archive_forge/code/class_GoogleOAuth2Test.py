import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class GoogleOAuth2Test(AsyncHTTPTestCase):

    def get_app(self):
        return Application([('/client/login', GoogleLoginHandler, dict(test=self)), ('/google/oauth2/authorize', GoogleOAuth2AuthorizeHandler), ('/google/oauth2/token', GoogleOAuth2TokenHandler), ('/google/oauth2/userinfo', GoogleOAuth2UserinfoHandler)], google_oauth={'key': 'fake_google_client_id', 'secret': 'fake_google_client_secret'})

    def test_google_login(self):
        response = self.fetch('/client/login')
        self.assertDictEqual({'name': 'Foo', 'email': 'foo@example.com', 'access_token': 'fake-access-token'}, json_decode(response.body))