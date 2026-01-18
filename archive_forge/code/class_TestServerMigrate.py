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
class TestServerMigrate(TestServer):

    def setUp(self):
        super(TestServerMigrate, self).setUp()
        methods = {'migrate': None, 'live_migrate': None}
        self.server = compute_fakes.create_one_server(methods=methods)
        self.servers_mock.get.return_value = self.server
        self.servers_mock.migrate.return_value = None
        self.servers_mock.live_migrate.return_value = None
        self.cmd = server.MigrateServer(self.app, None)

    def test_server_migrate_no_options(self):
        arglist = [self.server.id]
        verifylist = [('live_migration', False), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.migrate.assert_called_with()
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertIsNone(result)

    def test_server_migrate_with_host_2_56(self):
        arglist = ['--host', 'fakehost', self.server.id]
        verifylist = [('live_migration', False), ('host', 'fakehost'), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_client.api_version = api_versions.APIVersion('2.56')
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.migrate.assert_called_with(host='fakehost')
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertIsNone(result)

    def test_server_migrate_with_block_migration(self):
        arglist = ['--block-migration', self.server.id]
        verifylist = [('live_migration', False), ('block_migration', True), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertNotCalled(self.servers_mock.migrate)

    def test_server_migrate_with_disk_overcommit(self):
        arglist = ['--disk-overcommit', self.server.id]
        verifylist = [('live_migration', False), ('block_migration', None), ('disk_overcommit', True), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertNotCalled(self.servers_mock.migrate)

    def test_server_migrate_with_host_pre_v256(self):
        arglist = ['--host', 'fakehost', self.server.id]
        verifylist = [('live_migration', False), ('host', 'fakehost'), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.56 or greater is required to use --host without --live-migration.', str(ex))
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertNotCalled(self.servers_mock.migrate)

    def test_server_live_migrate(self):
        arglist = ['--live-migration', self.server.id]
        verifylist = [('live_migration', True), ('host', None), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.live_migrate.assert_called_with(block_migration=False, disk_over_commit=False, host=None)
        self.assertNotCalled(self.servers_mock.migrate)
        self.assertIsNone(result)

    def test_server_live_migrate_with_host(self):
        arglist = ['--live-migration', '--host', 'fakehost', self.server.id]
        verifylist = [('live_migration', True), ('host', 'fakehost'), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_client.api_version = api_versions.APIVersion('2.30')
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.live_migrate.assert_called_with(block_migration='auto', host='fakehost')
        self.assertNotCalled(self.servers_mock.migrate)
        self.assertIsNone(result)

    def test_server_live_migrate_with_host_pre_v230(self):
        arglist = ['--live-migration', '--host', 'fakehost', self.server.id]
        verifylist = [('live_migration', True), ('host', 'fakehost'), ('block_migration', None), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.30 or greater is required when using --host', str(ex))
        self.servers_mock.get.assert_called_with(self.server.id)
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertNotCalled(self.servers_mock.migrate)

    def test_server_block_live_migrate(self):
        arglist = ['--live-migration', '--block-migration', self.server.id]
        verifylist = [('live_migration', True), ('block_migration', True), ('disk_overcommit', None), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_client.api_version = api_versions.APIVersion('2.24')
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.live_migrate.assert_called_with(block_migration=True, disk_over_commit=False, host=None)
        self.assertNotCalled(self.servers_mock.migrate)
        self.assertIsNone(result)

    def test_server_live_migrate_with_disk_overcommit(self):
        arglist = ['--live-migration', '--disk-overcommit', self.server.id]
        verifylist = [('live_migration', True), ('block_migration', None), ('disk_overcommit', True), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_client.api_version = api_versions.APIVersion('2.24')
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.live_migrate.assert_called_with(block_migration=False, disk_over_commit=True, host=None)
        self.assertNotCalled(self.servers_mock.migrate)
        self.assertIsNone(result)

    def test_server_live_migrate_with_disk_overcommit_post_v224(self):
        arglist = ['--live-migration', '--disk-overcommit', self.server.id]
        verifylist = [('live_migration', True), ('block_migration', None), ('disk_overcommit', True), ('wait', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_client.api_version = api_versions.APIVersion('2.25')
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.live_migrate.assert_called_with(block_migration='auto', host=None)
        self.assertNotCalled(self.servers_mock.migrate)
        self.assertIsNone(result)
        mock_warning.assert_called_once()
        self.assertIn('The --disk-overcommit and --no-disk-overcommit options ', str(mock_warning.call_args[0][0]))

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_server_migrate_with_wait(self, mock_wait_for_status):
        arglist = ['--wait', self.server.id]
        verifylist = [('live_migration', False), ('block_migration', None), ('disk_overcommit', None), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.migrate.assert_called_with()
        self.assertNotCalled(self.servers_mock.live_migrate)
        self.assertIsNone(result)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_server_migrate_with_wait_fails(self, mock_wait_for_status):
        arglist = ['--wait', self.server.id]
        verifylist = [('live_migration', False), ('block_migration', None), ('disk_overcommit', None), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.migrate.assert_called_with()
        self.assertNotCalled(self.servers_mock.live_migrate)