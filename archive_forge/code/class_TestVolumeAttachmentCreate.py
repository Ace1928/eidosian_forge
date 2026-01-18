from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
class TestVolumeAttachmentCreate(TestVolumeAttachment):
    volume = volume_fakes.create_one_volume()
    server = compute_fakes.create_one_server()
    volume_attachment = volume_fakes.create_one_volume_attachment(attrs={'instance': server.id, 'volume_id': volume.id})
    columns = ('ID', 'Volume ID', 'Instance ID', 'Status', 'Attach Mode', 'Attached At', 'Detached At', 'Properties')
    data = (volume_attachment.id, volume_attachment.volume_id, volume_attachment.instance, volume_attachment.status, volume_attachment.attach_mode, volume_attachment.attached_at, volume_attachment.detached_at, format_columns.DictColumn(volume_attachment.connection_info))

    def setUp(self):
        super().setUp()
        self.volumes_mock.get.return_value = self.volume
        self.servers_mock.get.return_value = self.server
        self.volume_attachments_mock.create.return_value = self.volume_attachment.to_dict()
        self.cmd = volume_attachment.CreateVolumeAttachment(self.app, None)

    def test_volume_attachment_create(self):
        self.volume_client.api_version = api_versions.APIVersion('3.27')
        arglist = [self.volume.id, self.server.id]
        verifylist = [('volume', self.volume.id), ('server', self.server.id), ('connect', False), ('initiator', None), ('ip', None), ('host', None), ('platform', None), ('os_type', None), ('multipath', False), ('mountpoint', None), ('mode', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volumes_mock.get.assert_called_once_with(self.volume.id)
        self.servers_mock.get.assert_called_once_with(self.server.id)
        self.volume_attachments_mock.create.assert_called_once_with(self.volume.id, {}, self.server.id, None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_attachment_create_with_connect(self):
        self.volume_client.api_version = api_versions.APIVersion('3.54')
        arglist = [self.volume.id, self.server.id, '--connect', '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec', '--ip', '192.168.1.20', '--host', 'my-host', '--platform', 'x86_64', '--os-type', 'linux2', '--multipath', '--mountpoint', '/dev/vdb', '--mode', 'null']
        verifylist = [('volume', self.volume.id), ('server', self.server.id), ('connect', True), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb'), ('mode', 'null')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        connect_info = dict([('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb')])
        self.volumes_mock.get.assert_called_once_with(self.volume.id)
        self.servers_mock.get.assert_called_once_with(self.server.id)
        self.volume_attachments_mock.create.assert_called_once_with(self.volume.id, connect_info, self.server.id, 'null')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_attachment_create_pre_v327(self):
        self.volume_client.api_version = api_versions.APIVersion('3.26')
        arglist = [self.volume.id, self.server.id]
        verifylist = [('volume', self.volume.id), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.27 or greater is required', str(exc))

    def test_volume_attachment_create_with_mode_pre_v354(self):
        self.volume_client.api_version = api_versions.APIVersion('3.53')
        arglist = [self.volume.id, self.server.id, '--mode', 'rw']
        verifylist = [('volume', self.volume.id), ('server', self.server.id), ('mode', 'rw')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.54 or greater is required', str(exc))

    def test_volume_attachment_create_with_connect_missing_arg(self):
        self.volume_client.api_version = api_versions.APIVersion('3.54')
        arglist = [self.volume.id, self.server.id, '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec']
        verifylist = [('volume', self.volume.id), ('server', self.server.id), ('connect', False), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('You must specify the --connect option for any', str(exc))