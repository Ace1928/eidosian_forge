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