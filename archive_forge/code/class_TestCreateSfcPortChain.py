from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestCreateSfcPortChain(fakes.TestNeutronClientOSCV2):
    _port_chain = fakes.FakeSfcPortChain.create_port_chain()
    columns = ('Chain Parameters', 'Description', 'Flow Classifiers', 'ID', 'Name', 'Port Pair Groups', 'Project')

    def get_data(self):
        return (self._port_chain['chain_parameters'], self._port_chain['description'], self._port_chain['flow_classifiers'], self._port_chain['id'], self._port_chain['name'], self._port_chain['port_pair_groups'], self._port_chain['project_id'])

    def setUp(self):
        super(TestCreateSfcPortChain, self).setUp()
        self.network.create_sfc_port_chain = mock.Mock(return_value=self._port_chain)
        self.data = self.get_data()
        self.cmd = sfc_port_chain.CreateSfcPortChain(self.app, self.namespace)

    def test_create_port_chain_default_options(self):
        arglist = [self._port_chain['name'], '--port-pair-group', self._port_chain['port_pair_groups']]
        verifylist = [('name', self._port_chain['name']), ('port_pair_groups', [self._port_chain['port_pair_groups']]), ('flow_classifiers', []), ('chain_parameters', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_chain.assert_called_once_with(**{'name': self._port_chain['name'], 'port_pair_groups': [self._port_chain['port_pair_groups']]})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_port_chain_all_options(self):
        arglist = ['--description', self._port_chain['description'], '--port-pair-group', self._port_chain['port_pair_groups'], self._port_chain['name'], '--flow-classifier', self._port_chain['flow_classifiers'], '--chain-parameters', 'correlation=mpls,symmetric=true']
        cp = {'correlation': 'mpls', 'symmetric': 'true'}
        verifylist = [('port_pair_groups', [self._port_chain['port_pair_groups']]), ('name', self._port_chain['name']), ('description', self._port_chain['description']), ('flow_classifiers', [self._port_chain['flow_classifiers']]), ('chain_parameters', [cp])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_chain.assert_called_once_with(**{'name': self._port_chain['name'], 'port_pair_groups': [self._port_chain['port_pair_groups']], 'description': self._port_chain['description'], 'flow_classifiers': [self._port_chain['flow_classifiers']], 'chain_parameters': cp})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)