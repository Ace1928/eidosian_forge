import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
class TestKeystoneFlaskUnrouted404(rest.RestfulTestCase):

    def setUp(self):
        super(TestKeystoneFlaskUnrouted404, self).setUp()
        self.public_app.app.error_handler_spec[None].pop(404)

    def test_unrouted_path_is_not_jsonified_404(self):
        with self.test_client() as c:
            path = '/{unrouted_path}'.format(unrouted_path=uuid.uuid4())
            resp = c.get(path, expected_status_code=404)
            self.assertIn('text/html', resp.headers['Content-Type'])
            self.assertTrue(b'404 Not Found' in resp.data)