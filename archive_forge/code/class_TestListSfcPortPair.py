from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestListSfcPortPair(fakes.TestNeutronClientOSCV2):
    _port_pairs = fakes.FakeSfcPortPair.create_port_pairs()
    columns = ('ID', 'Name', 'Ingress Logical Port', 'Egress Logical Port')
    columns_long = ('ID', 'Name', 'Ingress Logical Port', 'Egress Logical Port', 'Service Function Parameters', 'Description', 'Project')
    _port_pair = _port_pairs[0]
    data = [_port_pair['id'], _port_pair['name'], _port_pair['ingress'], _port_pair['egress']]
    data_long = [_port_pair['id'], _port_pair['name'], _port_pair['ingress'], _port_pair['egress'], _port_pair['service_function_parameters'], _port_pair['description']]
    _port_pair1 = {'port_pairs': _port_pair}
    _port_pair_id = (_port_pair['id'],)

    def setUp(self):
        super(TestListSfcPortPair, self).setUp()
        self.network.sfc_port_pairs = mock.Mock(return_value=self._port_pairs)
        self.cmd = sfc_port_pair.ListSfcPortPair(self.app, self.namespace)

    def test_list_port_pairs(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns = self.cmd.take_action(parsed_args)[0]
        port_pairs = self.network.sfc_port_pairs()
        port_pair = port_pairs[0]
        data = [port_pair['id'], port_pair['name'], port_pair['ingress'], port_pair['egress']]
        self.assertEqual(list(self.columns), columns)
        self.assertEqual(self.data, data)

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        port_pairs = self.network.sfc_port_pairs()
        port_pair = port_pairs[0]
        data = [port_pair['id'], port_pair['name'], port_pair['ingress'], port_pair['egress'], port_pair['service_function_parameters'], port_pair['description']]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns_long = self.cmd.take_action(parsed_args)[0]
        self.assertEqual(list(self.columns_long), columns_long)
        self.assertEqual(self.data_long, data)