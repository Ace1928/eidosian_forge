import fixtures
import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import simple_cert
class SimpleCertRequestIdTests(utils.TestRequestId):

    def setUp(self):
        super(SimpleCertRequestIdTests, self).setUp()
        self.mgr = simple_cert.SimpleCertManager(self.client)

    def _mock_request_method(self, method=None, body=None):
        return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock

    def test_list_ca_certificates(self):
        body = {'certificates': [{'name': 'admin'}, {'name': 'admin2'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get_ca_certificates()
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('/OS-SIMPLE-CERT/ca', authenticated=False)

    def test_list_certificates(self):
        body = {'certificates': [{'name': 'admin'}, {'name': 'admin2'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get_certificates()
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with('/OS-SIMPLE-CERT/certificates', authenticated=False)