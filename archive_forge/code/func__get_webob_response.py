from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
def _get_webob_response(self):
    request = webob.Request.blank('/')
    response = webob.Response()
    response.request = request
    return response