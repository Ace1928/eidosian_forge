from unittest import mock
import uuid
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import vpnservice
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestVPNService(test_fakes.TestNeutronClientOSCV2):

    def _check_results(self, headers, data, exp_req, is_list=False):
        if is_list:
            req_body = {self.res_plural: list(exp_req)}
        else:
            req_body = exp_req
        self.mocked.assert_called_once_with(**req_body)
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)

    def setUp(self):
        super(TestVPNService, self).setUp()

        def _mock_vpnservice(*args, **kwargs):
            self.networkclient.find_vpn_service.assert_called_once_with(self.resource['id'], ignore_missing=False)
            return {'id': args[0]}
        self.networkclient.find_router = mock.Mock()
        self.networkclient.find_subnet = mock.Mock()
        self.fake_router = mock.Mock()
        self.fake_subnet = mock.Mock()
        self.networkclient.find_router.return_value = self.fake_router
        self.networkclient.find_subnet.return_value = self.fake_subnet
        self.args = {'name': 'my-name', 'description': 'my-desc', 'project_id': 'project-id-' + uuid.uuid4().hex, 'router_id': 'router-id-' + uuid.uuid4().hex, 'subnet_id': 'subnet-id-' + uuid.uuid4().hex}
        self.fake_subnet.id = self.args['subnet_id']
        self.fake_router.id = self.args['router_id']
        self.networkclient.find_vpn_service.side_effect = mock.Mock(side_effect=_mock_vpnservice)
        osc_utils.find_project = mock.Mock()
        osc_utils.find_project.id = _vpnservice['project_id']
        self.res = 'vpnservice'
        self.res_plural = 'vpnservices'
        self.resource = _vpnservice
        self.headers = ('ID', 'Name', 'Router', 'Subnet', 'Flavor', 'State', 'Status', 'Description', 'Project')
        self.data = _generate_data()
        self.ordered_headers = ('Description', 'Flavor', 'ID', 'Name', 'Project', 'Router', 'State', 'Status', 'Subnet')
        self.ordered_data = (_vpnservice['description'], _vpnservice['flavor_id'], _vpnservice['id'], _vpnservice['name'], _vpnservice['project_id'], _vpnservice['router_id'], _vpnservice['admin_state_up'], _vpnservice['status'], _vpnservice['subnet_id'])
        self.ordered_columns = ('description', 'flavor_id', 'id', 'name', 'project_id', 'router_id', 'admin_state_up', 'status', 'subnet_id')