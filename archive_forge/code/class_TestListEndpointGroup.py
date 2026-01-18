from unittest import mock
from osc_lib.tests import utils as tests_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import endpoint_group
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.vpnaas import common
from neutronclient.tests.unit.osc.v2.vpnaas import fakes
class TestListEndpointGroup(TestEndpointGroup):

    def setUp(self):
        super(TestListEndpointGroup, self).setUp()
        self.cmd = endpoint_group.ListEndpointGroup(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Type', 'Endpoints')
        self.short_data = (_endpoint_group['id'], _endpoint_group['name'], _endpoint_group['type'], _endpoint_group['endpoints'])
        self.networkclient.vpn_endpoint_groups = mock.Mock(return_value=[_endpoint_group])
        self.mocked = self.networkclient.vpn_endpoint_groups

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.headers), headers)

    def test_list_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))