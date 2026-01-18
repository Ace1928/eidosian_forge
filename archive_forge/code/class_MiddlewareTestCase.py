import sys
import time
from unittest import mock
import uuid
import warnings
import fixtures
from oslo_log import log as logging
import oslotest.base as oslotest
import requests
import webob
import webtest
class MiddlewareTestCase(BaseTestCase):

    def create_middleware(self, cb, **kwargs):
        raise NotImplementedError('implement this in your tests')

    def create_simple_middleware(self, status='200 OK', body='', headers=None, **kwargs):

        def cb(req):
            resp = webob.Response(body, status)
            resp.headers.update(headers or {})
            return resp
        return self.create_middleware(cb, **kwargs)

    def create_app(self, *args, **kwargs):
        return webtest.TestApp(self.create_middleware(*args, **kwargs))

    def create_simple_app(self, *args, **kwargs):
        return webtest.TestApp(self.create_simple_middleware(*args, **kwargs))