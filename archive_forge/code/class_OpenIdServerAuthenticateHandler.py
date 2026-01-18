import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OpenIdServerAuthenticateHandler(RequestHandler):

    def post(self):
        if self.get_argument('openid.mode') != 'check_authentication':
            raise Exception('incorrect openid.mode %r')
        self.write('is_valid:true')