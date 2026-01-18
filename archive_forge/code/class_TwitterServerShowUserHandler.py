import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class TwitterServerShowUserHandler(RequestHandler):

    def get(self, screen_name):
        if screen_name == 'error':
            raise HTTPError(500)
        assert 'oauth_nonce' in self.request.arguments
        assert 'oauth_timestamp' in self.request.arguments
        assert 'oauth_signature' in self.request.arguments
        assert self.get_argument('oauth_consumer_key') == 'test_twitter_consumer_key'
        assert self.get_argument('oauth_signature_method') == 'HMAC-SHA1'
        assert self.get_argument('oauth_version') == '1.0'
        assert self.get_argument('oauth_token') == 'hjkl'
        self.write(dict(screen_name=screen_name, name=screen_name.capitalize()))