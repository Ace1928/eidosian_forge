from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowFloatingIPNetwork(TestFloatingIPNetwork):
    floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip()
    columns = ('description', 'dns_domain', 'dns_name', 'fixed_ip_address', 'floating_ip_address', 'floating_network_id', 'id', 'port_id', 'project_id', 'qos_policy_id', 'router_id', 'status', 'tags')
    data = (floating_ip.description, floating_ip.dns_domain, floating_ip.dns_name, floating_ip.fixed_ip_address, floating_ip.floating_ip_address, floating_ip.floating_network_id, floating_ip.id, floating_ip.port_id, floating_ip.project_id, floating_ip.qos_policy_id, floating_ip.router_id, floating_ip.status, floating_ip.tags)

    def setUp(self):
        super(TestShowFloatingIPNetwork, self).setUp()
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.cmd = fip.ShowFloatingIP(self.app, self.namespace)

    def test_floating_ip_show(self):
        arglist = [self.floating_ip.id]
        verifylist = [('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)