from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as v2_volume_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_manage
class TestBlockStorageSnapshotManage(TestBlockStorageManage):
    snapshot_manage_list = volume_fakes.create_snapshot_manage_list_records()

    def setUp(self):
        super().setUp()
        self.snapshots_mock.list_manageable.return_value = self.snapshot_manage_list
        self.cmd = block_storage_manage.BlockStorageManageSnapshots(self.app, None)

    def test_block_storage_snapshot_manage_list(self):
        self.volume_client.api_version = api_versions.APIVersion('3.8')
        arglist = ['fake_host']
        verifylist = [('host', 'fake_host')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['reference', 'size', 'safe_to_manage', 'source_reference']
        datalist = []
        for snapshot_record in self.snapshot_manage_list:
            manage_details = (snapshot_record.reference, snapshot_record.size, snapshot_record.safe_to_manage, snapshot_record.source_reference)
            datalist.append(manage_details)
        datalist = tuple(datalist)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(datalist, tuple(data))
        self.snapshots_mock.list_manageable.assert_called_with(host='fake_host', detailed=False, marker=None, limit=None, offset=None, sort=None, cluster=None)

    def test_block_storage_snapshot_manage_list__pre_v38(self):
        arglist = ['fake_host']
        verifylist = [('host', 'fake_host')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.8 or greater is required', str(exc))

    def test_block_storage_snapshot_manage_list__pre_v317(self):
        self.volume_client.api_version = api_versions.APIVersion('3.16')
        arglist = ['--cluster', 'fake_cluster']
        verifylist = [('cluster', 'fake_cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.17 or greater is required', str(exc))
        self.assertIn('--cluster', str(exc))

    def test_block_storage_snapshot_manage_list__host_and_cluster(self):
        self.volume_client.api_version = api_versions.APIVersion('3.17')
        arglist = ['fake_host', '--cluster', 'fake_cluster']
        verifylist = [('host', 'fake_host'), ('cluster', 'fake_cluster')]
        exc = self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertIn('argument --cluster: not allowed with argument <host>', str(exc))

    def test_block_storage_snapshot_manage_list__detailed(self):
        self.volume_client.api_version = api_versions.APIVersion('3.8')
        arglist = ['--detailed', 'True', 'fake_host']
        verifylist = [('host', 'fake_host'), ('detailed', 'True'), ('marker', None), ('limit', None), ('offset', None), ('sort', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['reference', 'size', 'safe_to_manage', 'source_reference', 'reason_not_safe', 'cinder_id', 'extra_info']
        datalist = []
        for snapshot_record in self.snapshot_manage_list:
            manage_details = (snapshot_record.reference, snapshot_record.size, snapshot_record.safe_to_manage, snapshot_record.source_reference, snapshot_record.reason_not_safe, snapshot_record.cinder_id, snapshot_record.extra_info)
            datalist.append(manage_details)
        datalist = tuple(datalist)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(datalist, tuple(data))
        self.snapshots_mock.list_manageable.assert_called_with(host='fake_host', detailed=True, marker=None, limit=None, offset=None, sort=None, cluster=None)
        mock_warning.assert_called_once()
        self.assertIn('The --detailed option has been deprecated.', str(mock_warning.call_args[0][0]))

    def test_block_storage_snapshot_manage_list__all_args(self):
        self.app.client_manager.volume.api_version = api_versions.APIVersion('3.8')
        arglist = ['--long', '--marker', 'fake_marker', '--limit', '5', '--offset', '3', '--sort', 'size:asc', 'fake_host']
        verifylist = [('host', 'fake_host'), ('detailed', None), ('long', True), ('marker', 'fake_marker'), ('limit', '5'), ('offset', '3'), ('sort', 'size:asc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['reference', 'size', 'safe_to_manage', 'source_reference', 'reason_not_safe', 'cinder_id', 'extra_info']
        datalist = []
        for snapshot_record in self.snapshot_manage_list:
            manage_details = (snapshot_record.reference, snapshot_record.size, snapshot_record.safe_to_manage, snapshot_record.source_reference, snapshot_record.reason_not_safe, snapshot_record.cinder_id, snapshot_record.extra_info)
            datalist.append(manage_details)
        datalist = tuple(datalist)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(datalist, tuple(data))
        self.snapshots_mock.list_manageable.assert_called_with(host='fake_host', detailed=True, marker='fake_marker', limit='5', offset='3', sort='size:asc', cluster=None)