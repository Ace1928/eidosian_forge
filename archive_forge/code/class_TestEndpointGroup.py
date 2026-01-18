from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestEndpointGroup(test_fakes.TestNeutronClientOSCV2):

    def check_results(self, headers, data, exp_req, is_list=False):
        if is_list:
            req_body = {self.res_plural: list(exp_req)}
        else:
            req_body = exp_req
        self.mocked.assert_called_once_with(**req_body)
        self.assertEqual(self.ordered_headers, tuple(sorted(headers)))
        self.assertEqual(self.ordered_data, data)

    def setUp(self):
        super(TestEndpointGroup, self).setUp()

        def _mock_endpoint_group(*args, **kwargs):
            self.networkclient.find_vpn_endpoint_group.assert_called_once_with(self.resource['id'], ignore_missing=False)
            return {'id': args[0]}
        self.networkclient.find_vpn_endpoint_group.side_effect = mock.Mock(side_effect=_mock_endpoint_group)
        osc_utils.find_project = mock.Mock()
        osc_utils.find_project.id = _endpoint_group['project_id']
        self.res = 'endpoint_group'
        self.res_plural = 'endpoint_groups'
        self.resource = _endpoint_group
        self.headers = ('ID', 'Name', 'Type', 'Endpoints', 'Description', 'Project')
        self.data = _generate_data()
        self.ordered_headers = ('Description', 'Endpoints', 'ID', 'Name', 'Project', 'Type')
        self.ordered_data = (_endpoint_group['description'], _endpoint_group['endpoints'], _endpoint_group['id'], _endpoint_group['name'], _endpoint_group['project_id'], _endpoint_group['type'])
        self.ordered_columns = ('description', 'endpoints', 'id', 'name', 'project_id', 'type')