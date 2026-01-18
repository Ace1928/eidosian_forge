from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowFloatingIPPortForwarding(TestFloatingIPPortForwarding):
    columns = ('description', 'external_port', 'external_port_range', 'floatingip_id', 'id', 'internal_ip_address', 'internal_port', 'internal_port_id', 'internal_port_range', 'protocol')

    def setUp(self):
        super(TestShowFloatingIPPortForwarding, self).setUp()
        self._port_forwarding = network_fakes.FakeFloatingIPPortForwarding.create_one_port_forwarding(attrs={'floatingip_id': self.floating_ip.id})
        self.data = (self._port_forwarding.description, self._port_forwarding.external_port, self._port_forwarding.external_port_range, self._port_forwarding.floatingip_id, self._port_forwarding.id, self._port_forwarding.internal_ip_address, self._port_forwarding.internal_port, self._port_forwarding.internal_port_id, self._port_forwarding.internal_port_range, self._port_forwarding.protocol)
        self.network_client.find_floating_ip_port_forwarding = mock.Mock(return_value=self._port_forwarding)
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.cmd = floating_ip_port_forwarding.ShowFloatingIPPortForwarding(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_default_options(self):
        arglist = [self._port_forwarding.floatingip_id, self._port_forwarding.id]
        verifylist = [('floating_ip', self._port_forwarding.floatingip_id), ('port_forwarding_id', self._port_forwarding.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_floating_ip_port_forwarding.assert_called_once_with(self.floating_ip, self._port_forwarding.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.data), list(data))