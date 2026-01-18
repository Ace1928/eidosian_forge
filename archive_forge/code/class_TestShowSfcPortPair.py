from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestShowSfcPortPair(fakes.TestNeutronClientOSCV2):
    _pp = fakes.FakeSfcPortPair.create_port_pair()
    data = (_pp['description'], _pp['egress'], _pp['id'], _pp['ingress'], _pp['name'], _pp['project_id'], _pp['service_function_parameters'])
    _port_pair = _pp
    _port_pair_id = _pp['id']
    columns = ('Description', 'Egress Logical Port', 'ID', 'Ingress Logical Port', 'Name', 'Project', 'Service Function Parameters')

    def setUp(self):
        super(TestShowSfcPortPair, self).setUp()
        self.network.get_sfc_port_pair = mock.Mock(return_value=self._port_pair)
        self.cmd = sfc_port_pair.ShowSfcPortPair(self.app, self.namespace)

    def test_show_port_pair(self):
        client = self.app.client_manager.network
        mock_port_pair_show = client.get_sfc_port_pair
        arglist = [self._port_pair_id]
        verifylist = [('port_pair', self._port_pair_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_port_pair_show.assert_called_once_with(self._port_pair_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)