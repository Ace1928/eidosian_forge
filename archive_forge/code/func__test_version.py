import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def _test_version(self, app_name):
    app = self.loadapp(app_name)
    client = TestClient(app)
    resp = client.get('/')
    self.assertEqual(300, resp.status_int)
    data = jsonutils.loads(resp.body)
    expected = VERSIONS_RESPONSE
    url_with_port = 'http://localhost:%s/v3/' % self.public_port
    for version in expected['versions']['values']:
        if version['id'].startswith('v3'):
            self._paste_in_port(version, url_with_port)
    self.assertIn('Location', resp.headers)
    self.assertEqual(url_with_port, resp.headers['Location'])
    self.assertThat(data, _VersionsEqual(expected))