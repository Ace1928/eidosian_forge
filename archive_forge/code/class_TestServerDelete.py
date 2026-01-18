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
class TestServerDelete(TestServer):

    def setUp(self):
        super(TestServerDelete, self).setUp()
        self.servers_mock.delete.return_value = None
        self.servers_mock.force_delete.return_value = None
        self.cmd = server.DeleteServer(self.app, None)

    def test_server_delete_no_options(self):
        servers = self.setup_servers_mock(count=1)
        arglist = [servers[0].id]
        verifylist = [('server', [servers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.delete.assert_called_with(servers[0].id)
        self.servers_mock.force_delete.assert_not_called()
        self.assertIsNone(result)

    def test_server_delete_with_force(self):
        servers = self.setup_servers_mock(count=1)
        arglist = [servers[0].id, '--force']
        verifylist = [('server', [servers[0].id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.force_delete.assert_called_with(servers[0].id)
        self.servers_mock.delete.assert_not_called()
        self.assertIsNone(result)

    def test_server_delete_multi_servers(self):
        servers = self.setup_servers_mock(count=3)
        arglist = []
        verifylist = []
        for s in servers:
            arglist.append(s.id)
        verifylist = [('server', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in servers:
            calls.append(call(s.id))
        self.servers_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    @mock.patch.object(common_utils, 'find_resource')
    def test_server_delete_with_all_projects(self, mock_find_resource):
        servers = self.setup_servers_mock(count=1)
        mock_find_resource.side_effect = compute_fakes.get_servers(servers, 0)
        arglist = [servers[0].id, '--all-projects']
        verifylist = [('server', [servers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock_find_resource.assert_called_once_with(mock.ANY, servers[0].id, all_tenants=True)

    @mock.patch.object(common_utils, 'wait_for_delete', return_value=True)
    def test_server_delete_wait_ok(self, mock_wait_for_delete):
        servers = self.setup_servers_mock(count=1)
        arglist = [servers[0].id, '--wait']
        verifylist = [('server', [servers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.delete.assert_called_with(servers[0].id)
        mock_wait_for_delete.assert_called_once_with(self.servers_mock, servers[0].id, callback=mock.ANY)
        self.assertIsNone(result)

    @mock.patch.object(common_utils, 'wait_for_delete', return_value=False)
    def test_server_delete_wait_fails(self, mock_wait_for_delete):
        servers = self.setup_servers_mock(count=1)
        arglist = [servers[0].id, '--wait']
        verifylist = [('server', [servers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.delete.assert_called_with(servers[0].id)
        mock_wait_for_delete.assert_called_once_with(self.servers_mock, servers[0].id, callback=mock.ANY)