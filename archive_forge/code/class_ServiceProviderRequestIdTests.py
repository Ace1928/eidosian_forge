import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
class ServiceProviderRequestIdTests(utils.TestRequestId):

    def setUp(self):
        super(ServiceProviderRequestIdTests, self).setUp()
        self.mgr = service_providers.ServiceProviderManager(self.client)

    def _mock_request_method(self, method=None, body=None):
        return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock

    def test_get_service_provider(self):
        body = {'service_provider': {'name': 'admin'}}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get('provider')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/service_providers/provider')

    def test_list_service_provider(self):
        body = {'service_providers': [{'name': 'admin'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.list()
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/service_providers?')

    def test_create_service_provider(self):
        body = {'service_provider': {'name': 'admin'}}
        self._mock_request_method(method='post', body=body)
        put_mock = self._mock_request_method(method='put', body=body)
        response = self.mgr.create(id='provider')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        put_mock.assert_called_once_with('OS-FEDERATION/service_providers/provider', body={'service_provider': {}})

    def test_update_service_provider(self):
        body = {'service_provider': {'name': 'admin'}}
        patch_mock = self._mock_request_method(method='patch', body=body)
        self._mock_request_method(method='post', body=body)
        response = self.mgr.update('provider')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        patch_mock.assert_called_once_with('OS-FEDERATION/service_providers/provider', body={'service_provider': {}})

    def test_delete_service_provider(self):
        get_mock = self._mock_request_method(method='delete')
        _, resp = self.mgr.delete('provider')
        self.assertEqual(resp.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('OS-FEDERATION/service_providers/provider')