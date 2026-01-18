import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class GenericAuthPluginTests(utils.TestCase):
    ENDPOINT_FILTER = {uuid.uuid4().hex: uuid.uuid4().hex}

    def setUp(self):
        super(GenericAuthPluginTests, self).setUp()
        self.auth = GenericPlugin()
        self.session = session.Session(auth=self.auth)

    def test_setting_headers(self):
        text = uuid.uuid4().hex
        self.stub_url('GET', base_url=self.auth.url('prefix'), text=text)
        resp = self.session.get('prefix', endpoint_filter=self.ENDPOINT_FILTER)
        self.assertEqual(text, resp.text)
        for k, v in self.auth.headers.items():
            self.assertRequestHeaderEqual(k, v)
        self.assertIsNone(self.session.get_token())
        self.assertEqual(self.auth.headers, self.session.get_auth_headers())
        self.assertNotIn('X-Auth-Token', self.requests_mock.last_request.headers)

    def test_setting_connection_params(self):
        text = uuid.uuid4().hex
        self.stub_url('GET', base_url=self.auth.url('prefix'), text=text)
        resp = self.session.get('prefix', endpoint_filter=self.ENDPOINT_FILTER)
        self.assertEqual(text, resp.text)
        self.assertEqual(self.auth.cert, self.requests_mock.last_request.cert)
        self.assertFalse(self.requests_mock.last_request.verify)

    def test_setting_bad_connection_params(self):
        name = uuid.uuid4().hex
        self.auth.connection_params[name] = uuid.uuid4().hex
        e = self.assertRaises(exceptions.UnsupportedParameters, self.session.get, 'prefix', endpoint_filter=self.ENDPOINT_FILTER)
        self.assertIn(name, str(e))