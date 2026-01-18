import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerAddFixedIP(TestServer):

    def setUp(self):
        super().setUp()
        self.cmd = server.AddFixedIP(self.app, None)
        self.find_network = mock.Mock()
        self.app.client_manager.network.find_network = self.find_network

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_fixed_ip_pre_v249_with_tag(self, sm_mock):
        sm_mock.side_effect = [False, True]
        servers = self.setup_sdk_servers_mock(count=1)
        network = compute_fakes.create_one_network()
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            arglist = [servers[0].id, network['id'], '--fixed-ip-address', '5.6.7.8', '--tag', 'tag1']
            verifylist = [('server', servers[0].id), ('network', network['id']), ('fixed_ip_address', '5.6.7.8'), ('tag', 'tag1')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
            self.assertIn('--os-compute-api-version 2.49 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_fixed_ip(self, sm_mock):
        sm_mock.side_effect = [True, False]
        servers = self.setup_sdk_servers_mock(count=1)
        network = compute_fakes.create_one_network()
        interface = compute_fakes.create_one_server_interface()
        self.compute_sdk_client.create_server_interface.return_value = interface
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            arglist = [servers[0].id, network['id']]
            verifylist = [('server', servers[0].id), ('network', network['id'])]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            expected_columns = ('Port ID', 'Server ID', 'Network ID', 'MAC Address', 'Port State', 'Fixed IPs')
            expected_data = (interface.port_id, interface.server_id, interface.net_id, interface.mac_addr, interface.port_state, format_columns.ListDictColumn(interface.fixed_ips))
            columns, data = self.cmd.take_action(parsed_args)
            self.assertEqual(expected_columns, columns)
            self.assertEqual(expected_data, tuple(data))
            self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0].id, net_id=network['id'])

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_fixed_ip_with_fixed_ip(self, sm_mock):
        sm_mock.side_effect = [True, True]
        servers = self.setup_sdk_servers_mock(count=1)
        network = compute_fakes.create_one_network()
        interface = compute_fakes.create_one_server_interface()
        self.compute_sdk_client.create_server_interface.return_value = interface
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            arglist = [servers[0].id, network['id'], '--fixed-ip-address', '5.6.7.8']
            verifylist = [('server', servers[0].id), ('network', network['id']), ('fixed_ip_address', '5.6.7.8')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            expected_columns = ('Port ID', 'Server ID', 'Network ID', 'MAC Address', 'Port State', 'Fixed IPs')
            expected_data = (interface.port_id, interface.server_id, interface.net_id, interface.mac_addr, interface.port_state, format_columns.ListDictColumn(interface.fixed_ips))
            columns, data = self.cmd.take_action(parsed_args)
            self.assertEqual(expected_columns, columns)
            self.assertEqual(expected_data, tuple(data))
            self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0].id, net_id=network['id'], fixed_ips=[{'ip_address': '5.6.7.8'}])

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_fixed_ip_with_tag(self, sm_mock):
        sm_mock.side_effect = [True, True, True]
        servers = self.setup_sdk_servers_mock(count=1)
        network = compute_fakes.create_one_network()
        interface = compute_fakes.create_one_server_interface()
        self.compute_sdk_client.create_server_interface.return_value = interface
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            arglist = [servers[0].id, network['id'], '--fixed-ip-address', '5.6.7.8', '--tag', 'tag1']
            verifylist = [('server', servers[0].id), ('network', network['id']), ('fixed_ip_address', '5.6.7.8'), ('tag', 'tag1')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            expected_columns = ('Port ID', 'Server ID', 'Network ID', 'MAC Address', 'Port State', 'Fixed IPs', 'Tag')
            expected_data = (interface.port_id, interface.server_id, interface.net_id, interface.mac_addr, interface.port_state, format_columns.ListDictColumn(interface.fixed_ips), interface.tag)
            columns, data = self.cmd.take_action(parsed_args)
            self.assertEqual(expected_columns, columns)
            self.assertEqual(expected_data, tuple(data))
            self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0].id, net_id=network['id'], fixed_ips=[{'ip_address': '5.6.7.8'}], tag='tag1')

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_add_fixed_ip_with_fixed_ip_with_tag(self, sm_mock):
        sm_mock.side_effect = [True, True]
        servers = self.setup_sdk_servers_mock(count=1)
        network = compute_fakes.create_one_network()
        interface = compute_fakes.create_one_server_interface()
        self.compute_sdk_client.create_server_interface.return_value = interface
        with mock.patch.object(self.app.client_manager, 'is_network_endpoint_enabled', return_value=False):
            arglist = [servers[0].id, network['id'], '--fixed-ip-address', '5.6.7.8', '--tag', 'tag1']
            verifylist = [('server', servers[0].id), ('network', network['id']), ('fixed_ip_address', '5.6.7.8'), ('tag', 'tag1')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            expected_columns = ('Port ID', 'Server ID', 'Network ID', 'MAC Address', 'Port State', 'Fixed IPs', 'Tag')
            expected_data = (interface.port_id, interface.server_id, interface.net_id, interface.mac_addr, interface.port_state, format_columns.ListDictColumn(interface.fixed_ips), interface.tag)
            columns, data = self.cmd.take_action(parsed_args)
            self.assertEqual(expected_columns, columns)
            self.assertEqual(expected_data, tuple(data))
            self.compute_sdk_client.create_server_interface.assert_called_once_with(servers[0].id, net_id=network['id'], fixed_ips=[{'ip_address': '5.6.7.8'}], tag='tag1')