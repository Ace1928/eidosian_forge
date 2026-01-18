from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListFloatingIPPortForwarding(TestFloatingIPPortForwarding):
    columns = ('ID', 'Internal Port ID', 'Internal IP Address', 'Internal Port', 'Internal Port Range', 'External Port', 'External Port Range', 'Protocol', 'Description')

    def setUp(self):
        super(TestListFloatingIPPortForwarding, self).setUp()
        self.port_forwardings = network_fakes.FakeFloatingIPPortForwarding.create_port_forwardings(count=3, attrs={'internal_port_id': self.port.id, 'floatingip_id': self.floating_ip.id})
        self.data = []
        for port_forwarding in self.port_forwardings:
            self.data.append((port_forwarding.id, port_forwarding.internal_port_id, port_forwarding.internal_ip_address, port_forwarding.internal_port, port_forwarding.internal_port_range, port_forwarding.external_port, port_forwarding.external_port_range, port_forwarding.protocol, port_forwarding.description))
        self.network_client.floating_ip_port_forwardings = mock.Mock(return_value=self.port_forwardings)
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.cmd = floating_ip_port_forwarding.ListFloatingIPPortForwarding(self.app, self.namespace)

    def test_port_forwarding_list(self):
        arglist = [self.floating_ip.id]
        verifylist = [('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.floating_ip_port_forwardings.assert_called_once_with(self.floating_ip, **{})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_port_forwarding_list_all_options(self):
        arglist = ['--port', self.port_forwardings[0].internal_port_id, '--external-protocol-port', str(self.port_forwardings[0].external_port), '--protocol', self.port_forwardings[0].protocol, self.port_forwardings[0].floatingip_id]
        verifylist = [('port', self.port_forwardings[0].internal_port_id), ('external_protocol_port', str(self.port_forwardings[0].external_port)), ('protocol', self.port_forwardings[0].protocol), ('floating_ip', self.port_forwardings[0].floatingip_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        query = {'internal_port_id': self.port_forwardings[0].internal_port_id, 'external_port': self.port_forwardings[0].external_port, 'protocol': self.port_forwardings[0].protocol}
        self.network_client.floating_ip_port_forwardings.assert_called_once_with(self.floating_ip, **query)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))