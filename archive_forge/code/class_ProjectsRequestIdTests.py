import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
class ProjectsRequestIdTests(utils.TestRequestId):
    url = '/projects'

    def setUp(self):
        super(ProjectsRequestIdTests, self).setUp()
        self.mgr = projects.ProjectManager(self.client)
        self.mgr.resource_class = projects.Project

    def _mock_request_method(self, method=None, body=None):
        return self.useFixture(fixtures.MockPatchObject(self.client, method, autospec=True, return_value=(self.resp, body))).mock

    def test_get_project(self):
        body = {'project': {'name': 'admin'}}
        get_mock = self._mock_request_method(method='get', body=body)
        response = self.mgr.get(project='admin')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with(self.url + '/admin')

    def test_create_project(self):
        body = {'project': {'name': 'admin', 'domain': 'admin'}}
        post_mock = self._mock_request_method(method='post', body=body)
        response = self.mgr.create('admin', 'admin')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        post_mock.assert_called_once_with(self.url, body={'project': {'name': 'admin', 'enabled': True, 'domain_id': 'admin'}})

    def test_list_project(self):
        body = {'projects': [{'name': 'admin'}, {'name': 'admin'}]}
        get_mock = self._mock_request_method(method='get', body=body)
        returned_list = self.mgr.list()
        self.assertEqual(returned_list.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with(self.url + '?')

    def test_update_project(self):
        body = {'project': {'name': 'admin'}}
        patch_mock = self._mock_request_method(method='patch', body=body)
        put_mock = self._mock_request_method(method='put', body=body)
        response = self.mgr.update('admin', domain='demo')
        self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
        patch_mock.assert_called_once_with(self.url + '/admin', body={'project': {'domain_id': 'demo'}})
        self.assertFalse(put_mock.called)

    def test_delete_project(self):
        get_mock = self._mock_request_method(method='delete')
        _, resp = self.mgr.delete('admin')
        self.assertEqual(resp.request_ids[0], self.TEST_REQUEST_ID)
        get_mock.assert_called_once_with(self.url + '/admin')