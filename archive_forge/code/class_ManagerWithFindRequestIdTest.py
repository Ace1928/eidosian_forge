import uuid
import fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
import requests
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
from keystoneclient import utils as base_utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import roles
from keystoneclient.v3 import users
class ManagerWithFindRequestIdTest(utils.TestCase):
    url = '/fakes'
    resp = create_response_with_request_id_header()

    def setUp(self):
        super(ManagerWithFindRequestIdTest, self).setUp()
        auth = v2.Token(auth_url='http://127.0.0.1:5000', token=self.TEST_TOKEN)
        session_ = session.Session(auth=auth)
        self.client = client.Client(session=session_, include_metadata='True')._adapter

    def test_find_resource(self):
        body = {'roles': [{'name': 'entity_one'}, {'name': 'entity_one_1'}]}
        request_resp = requests.Response()
        request_resp.headers['x-openstack-request-id'] = TEST_REQUEST_ID
        get_mock = self.useFixture(fixtures.MockPatchObject(self.client, 'get', autospec=True, side_effect=[exceptions.NotFound, (request_resp, body)])).mock
        mgr = roles.RoleManager(self.client)
        mgr.resource_class = roles.Role
        response = base_utils.find_resource(mgr, 'entity_one')
        get_mock.assert_called_with('/OS-KSADM/roles')
        self.assertEqual(response.request_ids[0], TEST_REQUEST_ID)