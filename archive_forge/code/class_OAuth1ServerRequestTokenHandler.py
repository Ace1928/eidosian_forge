import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OAuth1ServerRequestTokenHandler(RequestHandler):

    def get(self):
        self.write('oauth_token=zxcv&oauth_token_secret=1234')