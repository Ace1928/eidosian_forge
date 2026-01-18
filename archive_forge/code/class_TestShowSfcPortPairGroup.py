from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestShowSfcPortPairGroup(fakes.TestNeutronClientOSCV2):
    _ppg = fakes.FakeSfcPortPairGroup.create_port_pair_group()
    data = (_ppg['description'], _ppg['id'], _ppg['name'], _ppg['port_pairs'], _ppg['port_pair_group_parameters'], _ppg['project_id'], _ppg['tap_enabled'])
    _port_pair_group = _ppg
    _port_pair_group_id = _ppg['id']
    columns = ('Description', 'ID', 'Name', 'Port Pair', 'Port Pair Group Parameters', 'Project', 'Tap Enabled')

    def setUp(self):
        super(TestShowSfcPortPairGroup, self).setUp()
        self.network.get_sfc_port_pair_group = mock.Mock(return_value=self._port_pair_group)
        self.cmd = sfc_port_pair_group.ShowSfcPortPairGroup(self.app, self.namespace)

    def test_show_port_pair_group(self):
        client = self.app.client_manager.network
        mock_port_pair_group_show = client.get_sfc_port_pair_group
        arglist = [self._port_pair_group_id]
        verifylist = [('port_pair_group', self._port_pair_group_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_port_pair_group_show.assert_called_once_with(self._port_pair_group_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)