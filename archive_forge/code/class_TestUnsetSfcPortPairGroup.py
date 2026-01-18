from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestUnsetSfcPortPairGroup(fakes.TestNeutronClientOSCV2):
    _port_pair_group = fakes.FakeSfcPortPairGroup.create_port_pair_group()
    resource = _port_pair_group
    res = 'port_pair_group'
    _port_pair_group_name = _port_pair_group['name']
    _port_pair_group_id = _port_pair_group['id']
    ppg_pp = _port_pair_group['port_pairs']

    def setUp(self):
        super(TestUnsetSfcPortPairGroup, self).setUp()
        self.network.update_sfc_port_pair_group = mock.Mock(return_value=None)
        self.mocked = self.network.update_sfc_port_pair_group
        self.cmd = sfc_port_pair_group.UnsetSfcPortPairGroup(self.app, self.namespace)

    def test_unset_port_pair(self):
        target = self.resource['id']
        port_pair1 = 'additional_port1'
        port_pair2 = 'additional_port2'
        self.network.find_sfc_port_pair = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_sfc_port_pair_group = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'port_pairs': self.ppg_pp})
        arglist = [target, '--port-pair', port_pair1, '--port-pair', port_pair2]
        verifylist = [(self.res, target), ('port_pairs', [port_pair1, port_pair2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'port_pairs': sorted([*self.ppg_pp])}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertIsNone(result)

    def test_unset_all_port_pair(self):
        client = self.app.client_manager.network
        mock_port_pair_group_update = client.update_sfc_port_pair_group
        arglist = [self._port_pair_group_name, '--all-port-pair']
        verifylist = [('port_pair_group', self._port_pair_group_name), ('all_port_pair', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'port_pairs': []}
        mock_port_pair_group_update.assert_called_once_with(self._port_pair_group_name, **attrs)
        self.assertIsNone(result)