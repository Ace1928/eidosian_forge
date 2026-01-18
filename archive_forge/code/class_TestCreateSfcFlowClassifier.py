from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestCreateSfcFlowClassifier(fakes.TestNeutronClientOSCV2):
    _fc = fakes.FakeSfcFlowClassifier.create_flow_classifier()
    columns = ('Description', 'Destination IP', 'Destination Port Range Max', 'Destination Port Range Min', 'Ethertype', 'ID', 'L7 Parameters', 'Logical Destination Port', 'Logical Source Port', 'Name', 'Project', 'Protocol', 'Source IP', 'Source Port Range Max', 'Source Port Range Min', 'Summary')

    def get_data(self):
        return (self._fc['description'], self._fc['destination_ip_prefix'], self._fc['destination_port_range_max'], self._fc['destination_port_range_min'], self._fc['ethertype'], self._fc['id'], self._fc['l7_parameters'], self._fc['logical_destination_port'], self._fc['logical_source_port'], self._fc['name'], self._fc['project_id'], self._fc['protocol'], self._fc['source_ip_prefix'], self._fc['source_port_range_max'], self._fc['source_port_range_min'])

    def setUp(self):
        super(TestCreateSfcFlowClassifier, self).setUp()
        self.network.create_sfc_flow_classifier = mock.Mock(return_value=self._fc)
        self.data = self.get_data()
        self.cmd = sfc_flow_classifier.CreateSfcFlowClassifier(self.app, self.namespace)

    def test_create_flow_classifier_default_options(self):
        arglist = ['--logical-source-port', self._fc['logical_source_port'], '--ethertype', self._fc['ethertype'], self._fc['name']]
        verifylist = [('logical_source_port', self._fc['logical_source_port']), ('ethertype', self._fc['ethertype']), ('name', self._fc['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_flow_classifier.assert_called_once_with(**{'name': self._fc['name'], 'logical_source_port': self._fc['logical_source_port'], 'ethertype': self._fc['ethertype']})
        self.assertEqual(self.columns, columns)

    def test_create_flow_classifier(self):
        arglist = ['--description', self._fc['description'], '--ethertype', self._fc['ethertype'], '--protocol', self._fc['protocol'], '--source-ip-prefix', self._fc['source_ip_prefix'], '--destination-ip-prefix', self._fc['destination_ip_prefix'], '--logical-source-port', self._fc['logical_source_port'], '--logical-destination-port', self._fc['logical_destination_port'], self._fc['name'], '--l7-parameters', 'url=path']
        param = 'url=path'
        verifylist = [('description', self._fc['description']), ('name', self._fc['name']), ('ethertype', self._fc['ethertype']), ('protocol', self._fc['protocol']), ('source_ip_prefix', self._fc['source_ip_prefix']), ('destination_ip_prefix', self._fc['destination_ip_prefix']), ('logical_source_port', self._fc['logical_source_port']), ('logical_destination_port', self._fc['logical_destination_port']), ('l7_parameters', param)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_flow_classifier.assert_called_once_with(**{'name': self._fc['name'], 'description': self._fc['description'], 'ethertype': self._fc['ethertype'], 'protocol': self._fc['protocol'], 'source_ip_prefix': self._fc['source_ip_prefix'], 'destination_ip_prefix': self._fc['destination_ip_prefix'], 'logical_source_port': self._fc['logical_source_port'], 'logical_destination_port': self._fc['logical_destination_port'], 'l7_parameters': param})
        self.assertEqual(self.columns, columns)