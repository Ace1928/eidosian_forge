import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
class MiddlewareRequestTestBase(unit.TestCase):
    MIDDLEWARE_CLASS = None

    def _application(self):
        """A base wsgi application that returns a simple response."""

        def app(environ, start_response):
            body = uuid.uuid4().hex.encode('utf-8')
            resp_headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
            start_response('200 OK', resp_headers)
            return [body]
        return app

    def _generate_app_response(self, app, headers=None, method='get', path='/', **kwargs):
        """Given a wsgi application wrap it in webtest and call it."""
        return getattr(webtest.TestApp(app), method)(path, headers=headers or {}, **kwargs)

    def _middleware_failure(self, exc, *args, **kwargs):
        """Assert that an exception is being thrown from process_request."""

        class _Failing(self.MIDDLEWARE_CLASS):
            _called = False

            def fill_context(i_self, *i_args, **i_kwargs):
                e = self.assertRaises(exc, super(_Failing, i_self).fill_context, *i_args, **i_kwargs)
                i_self._called = True
                raise e
        kwargs.setdefault('status', http.client.INTERNAL_SERVER_ERROR)
        app = _Failing(self._application())
        resp = self._generate_app_response(app, *args, **kwargs)
        self.assertTrue(app._called)
        return resp

    def _do_middleware_response(self, *args, **kwargs):
        """Wrap a middleware around a sample application and call it."""
        app = self.MIDDLEWARE_CLASS(self._application())
        return self._generate_app_response(app, *args, **kwargs)

    def _do_middleware_request(self, *args, **kwargs):
        """The request object from a successful middleware call."""
        return self._do_middleware_response(*args, **kwargs).request