from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestCreateSfcPortPair(fakes.TestNeutronClientOSCV2):
    _port_pair = fakes.FakeSfcPortPair.create_port_pair()
    columns = ('Description', 'Egress Logical Port', 'ID', 'Ingress Logical Port', 'Name', 'Project', 'Service Function Parameters')

    def get_data(self):
        return (self._port_pair['description'], self._port_pair['egress'], self._port_pair['id'], self._port_pair['ingress'], self._port_pair['name'], self._port_pair['project_id'], self._port_pair['service_function_parameters'])

    def setUp(self):
        super(TestCreateSfcPortPair, self).setUp()
        self.network.create_sfc_port_pair = mock.Mock(return_value=self._port_pair)
        self.data = self.get_data()
        self.cmd = sfc_port_pair.CreateSfcPortPair(self.app, self.namespace)

    def test_create_port_pair_default_options(self):
        arglist = ['--ingress', self._port_pair['ingress'], '--egress', self._port_pair['egress'], self._port_pair['name']]
        verifylist = [('ingress', self._port_pair['ingress']), ('egress', self._port_pair['egress']), ('name', self._port_pair['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_pair.assert_called_once_with(**{'name': self._port_pair['name'], 'ingress': self._port_pair['ingress'], 'egress': self._port_pair['egress']})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def _test_create_port_pair_all_options(self, correlation):
        arglist = ['--description', self._port_pair['description'], '--egress', self._port_pair['egress'], '--ingress', self._port_pair['ingress'], self._port_pair['name'], '--service-function-parameters', 'correlation=%s,weight=1' % correlation]
        verifylist = [('ingress', self._port_pair['ingress']), ('egress', self._port_pair['egress']), ('name', self._port_pair['name']), ('description', self._port_pair['description']), ('service_function_parameters', [{'correlation': correlation, 'weight': '1'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        if correlation == 'None':
            correlation_param = None
        else:
            correlation_param = correlation
        self.network.create_sfc_port_pair.assert_called_once_with(**{'name': self._port_pair['name'], 'ingress': self._port_pair['ingress'], 'egress': self._port_pair['egress'], 'description': self._port_pair['description'], 'service_function_parameters': {'correlation': correlation_param, 'weight': '1'}})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_port_pair_all_options(self):
        self._test_create_port_pair_all_options('None')

    def test_create_port_pair_all_options_mpls(self):
        self._test_create_port_pair_all_options('mpls')