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
class TestServerReboot(TestServer):

    def setUp(self):
        super().setUp()
        self.compute_sdk_client.reboot_server.return_value = None
        self.cmd = server.RebootServer(self.app, None)

    def test_server_reboot(self):
        servers = self.setup_sdk_servers_mock(count=1)
        arglist = [servers[0].id]
        verifylist = [('server', servers[0].id), ('reboot_type', 'SOFT'), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.reboot_server.assert_called_once_with(servers[0].id, 'SOFT')
        self.assertIsNone(result)

    def test_server_reboot_with_hard(self):
        servers = self.setup_sdk_servers_mock(count=1)
        arglist = ['--hard', servers[0].id]
        verifylist = [('server', servers[0].id), ('reboot_type', 'HARD'), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.reboot_server.assert_called_once_with(servers[0].id, 'HARD')
        self.assertIsNone(result)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_server_reboot_with_wait(self, mock_wait_for_status):
        servers = self.setup_sdk_servers_mock(count=1)
        arglist = ['--wait', servers[0].id]
        verifylist = [('server', servers[0].id), ('reboot_type', 'SOFT'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.reboot_server.assert_called_once_with(servers[0].id, 'SOFT')
        mock_wait_for_status.assert_called_once_with(self.compute_sdk_client.get_server, servers[0].id, callback=mock.ANY)

    @mock.patch.object(server.LOG, 'error')
    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_server_reboot_with_wait_fails(self, mock_wait_for_status, mock_log):
        servers = self.setup_sdk_servers_mock(count=1)
        arglist = ['--wait', servers[0].id]
        verifylist = [('server', servers[0].id), ('reboot_type', 'SOFT'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.compute_sdk_client.reboot_server.assert_called_once_with(servers[0].id, 'SOFT')
        mock_wait_for_status.assert_called_once_with(self.compute_sdk_client.get_server, servers[0].id, callback=mock.ANY)