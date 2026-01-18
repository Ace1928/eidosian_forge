from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestUnsetSfcPortChain(fakes.TestNeutronClientOSCV2):
    _port_chain = fakes.FakeSfcPortChain.create_port_chain()
    resource = _port_chain
    res = 'port_chain'
    _port_chain_name = _port_chain['name']
    _port_chain_id = _port_chain['id']
    pc_ppg = _port_chain['port_pair_groups']
    pc_fc = _port_chain['flow_classifiers']

    def setUp(self):
        super(TestUnsetSfcPortChain, self).setUp()
        self.network.update_sfc_port_chain = mock.Mock(return_value=None)
        self.mocked = self.network.update_sfc_port_chain
        self.cmd = sfc_port_chain.UnsetSfcPortChain(self.app, self.namespace)

    def test_unset_port_pair_group(self):
        target = self.resource['id']
        ppg1 = 'port_pair_group1'
        self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'port_pair_groups': [self.pc_ppg]})
        self.network.find_sfc_port_pair_group.side_effect = lambda name_or_id, ignore_missing=False: {'id': name_or_id}
        arglist = [target, '--port-pair-group', ppg1]
        verifylist = [(self.res, target), ('port_pair_groups', [ppg1])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'port_pair_groups': [self.pc_ppg]}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertIsNone(result)

    def test_unset_flow_classifier(self):
        target = self.resource['id']
        fc1 = 'flow_classifier1'
        self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'flow_classifiers': [self.pc_fc]})
        self.network.find_sfc_flow_classifier.side_effect = lambda name_or_id, ignore_missing=False: {'id': name_or_id}
        arglist = [target, '--flow-classifier', fc1]
        verifylist = [(self.res, target), ('flow_classifiers', [fc1])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'flow_classifiers': [self.pc_fc]}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertIsNone(result)

    def test_unset_all_flow_classifier(self):
        client = self.app.client_manager.network
        target = self.resource['id']
        mock_port_chain_update = client.update_sfc_port_chain
        arglist = [target, '--all-flow-classifier']
        verifylist = [(self.res, target), ('all_flow_classifier', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'flow_classifiers': []}
        mock_port_chain_update.assert_called_once_with(target, **expect)
        self.assertIsNone(result)