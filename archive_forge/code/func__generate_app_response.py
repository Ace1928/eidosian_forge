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
def _generate_app_response(self, app, headers=None, method='get', path='/', **kwargs):
    """Given a wsgi application wrap it in webtest and call it."""
    return getattr(webtest.TestApp(app), method)(path, headers=headers or {}, **kwargs)