import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def _middleware_request(self, token, extra_environ=None):

    def application(environ, start_response):
        body = b'body'
        headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
        start_response('200 OK', headers)
        return [body]
    app = webtest.TestApp(auth_context.AuthContextMiddleware(application), extra_environ=extra_environ)
    resp = app.get('/', headers={authorization.AUTH_TOKEN_HEADER: token})
    self.assertEqual(b'body', resp.body)
    return resp.request