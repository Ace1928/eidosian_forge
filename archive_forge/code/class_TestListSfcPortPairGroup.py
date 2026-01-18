from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestListSfcPortPairGroup(fakes.TestNeutronClientOSCV2):
    _ppgs = fakes.FakeSfcPortPairGroup.create_port_pair_groups(count=1)
    columns = ('ID', 'Name', 'Port Pair', 'Port Pair Group Parameters', 'Tap Enabled')
    columns_long = ('ID', 'Name', 'Port Pair', 'Port Pair Group Parameters', 'Description', 'Project', 'Tap Enabled')
    _port_pair_group = _ppgs[0]
    data = [_port_pair_group['id'], _port_pair_group['name'], _port_pair_group['port_pairs'], _port_pair_group['port_pair_group_parameters'], _port_pair_group['tap_enabled']]
    data_long = [_port_pair_group['id'], _port_pair_group['name'], _port_pair_group['port_pairs'], _port_pair_group['port_pair_group_parameters'], _port_pair_group['description'], _port_pair_group['tap_enabled']]
    _port_pair_group1 = {'port_pair_groups': _port_pair_group}
    _port_pair_id = _port_pair_group['id']

    def setUp(self):
        super(TestListSfcPortPairGroup, self).setUp()
        self.network.sfc_port_pair_groups = mock.Mock(return_value=self._ppgs)
        self.cmd = sfc_port_pair_group.ListSfcPortPairGroup(self.app, self.namespace)

    def test_list_port_pair_groups(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns = self.cmd.take_action(parsed_args)[0]
        ppgs = self.network.sfc_port_pair_groups()
        ppg = ppgs[0]
        data = [ppg['id'], ppg['name'], ppg['port_pairs'], ppg['port_pair_group_parameters'], ppg['tap_enabled']]
        self.assertEqual(list(self.columns), columns)
        self.assertEqual(self.data, data)

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        ppgs = self.network.sfc_port_pair_groups()
        ppg = ppgs[0]
        data = [ppg['id'], ppg['name'], ppg['port_pairs'], ppg['port_pair_group_parameters'], ppg['description'], ppg['tap_enabled']]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns_long = self.cmd.take_action(parsed_args)[0]
        self.assertEqual(list(self.columns_long), columns_long)
        self.assertEqual(self.data_long, data)