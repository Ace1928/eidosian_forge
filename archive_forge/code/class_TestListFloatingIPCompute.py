from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.floating_ip_list')
class TestListFloatingIPCompute(compute_fakes.TestComputev2):
    _floating_ips = compute_fakes.create_floating_ips(count=3)
    columns = ('ID', 'Floating IP Address', 'Fixed IP Address', 'Server', 'Pool')
    data = []
    for ip in _floating_ips:
        data.append((ip['id'], ip['ip'], ip['fixed_ip'], ip['instance_id'], ip['pool']))

    def setUp(self):
        super(TestListFloatingIPCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = fip.ListFloatingIP(self.app, None)

    def test_floating_ip_list(self, fip_mock):
        fip_mock.return_value = self._floating_ips
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        fip_mock.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))