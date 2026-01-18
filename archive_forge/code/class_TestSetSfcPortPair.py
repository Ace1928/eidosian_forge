from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestSetSfcPortPair(fakes.TestNeutronClientOSCV2):
    _port_pair = fakes.FakeSfcPortPair.create_port_pair()
    _port_pair_name = _port_pair['name']
    _port_pair_id = _port_pair['id']

    def setUp(self):
        super(TestSetSfcPortPair, self).setUp()
        self.network.update_sfc_port_pair = mock.Mock(return_value=None)
        self.cmd = sfc_port_pair.SetSfcPortPair(self.app, self.namespace)

    def test_set_port_pair(self):
        client = self.app.client_manager.network
        mock_port_pair_update = client.update_sfc_port_pair
        arglist = [self._port_pair_name, '--name', 'name_updated', '--description', 'desc_updated']
        verifylist = [('port_pair', self._port_pair_name), ('name', 'name_updated'), ('description', 'desc_updated')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'name_updated', 'description': 'desc_updated'}
        mock_port_pair_update.assert_called_once_with(self._port_pair_name, **attrs)
        self.assertIsNone(result)