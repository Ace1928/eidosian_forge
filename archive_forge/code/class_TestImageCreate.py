import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestImageCreate(TestImage):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()

    def setUp(self):
        super().setUp()
        self.new_image = image_fakes.create_one_image()
        self.image_client.create_image.return_value = self.new_image
        self.image_client.update_image.return_value = self.new_image
        self.project_mock.get.return_value = self.project
        self.domain_mock.get.return_value = self.domain
        self.expected_columns, self.expected_data = zip(*sorted(_image._format_image(self.new_image).items()))
        self.cmd = _image.CreateImage(self.app, None)

    @mock.patch('sys.stdin', side_effect=[None])
    def test_image_reserve_no_options(self, raw_input):
        arglist = [self.new_image.name]
        verifylist = [('container_format', _image.DEFAULT_CONTAINER_FORMAT), ('disk_format', _image.DEFAULT_DISK_FORMAT), ('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT)
        self.assertEqual(self.expected_columns, columns)
        self.assertCountEqual(self.expected_data, data)

    @mock.patch('sys.stdin', side_effect=[None])
    def test_image_reserve_options(self, raw_input):
        arglist = ['--container-format', 'ovf', '--disk-format', 'ami', '--min-disk', '10', '--min-ram', '4', '--protected' if self.new_image.is_protected else '--unprotected', '--private' if self.new_image.visibility == 'private' else '--public', '--project', self.new_image.owner_id, '--project-domain', self.domain.id, self.new_image.name]
        verifylist = [('container_format', 'ovf'), ('disk_format', 'ami'), ('min_disk', 10), ('min_ram', 4), ('is_protected', self.new_image.is_protected), ('visibility', self.new_image.visibility), ('project', self.new_image.owner_id), ('project_domain', self.domain.id), ('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format='ovf', disk_format='ami', min_disk=10, min_ram=4, owner_id=self.project.id, is_protected=self.new_image.is_protected, visibility=self.new_image.visibility)
        self.assertEqual(self.expected_columns, columns)
        self.assertCountEqual(self.expected_data, data)

    def test_image_create_with_unexist_project(self):
        self.project_mock.get.side_effect = exceptions.NotFound(None)
        self.project_mock.find.side_effect = exceptions.NotFound(None)
        arglist = ['--container-format', 'ovf', '--disk-format', 'ami', '--min-disk', '10', '--min-ram', '4', '--protected', '--private', '--project', 'unexist_owner', 'graven']
        verifylist = [('container_format', 'ovf'), ('disk_format', 'ami'), ('min_disk', 10), ('min_ram', 4), ('is_protected', True), ('visibility', 'private'), ('project', 'unexist_owner'), ('name', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_image_create_file(self):
        imagefile = tempfile.NamedTemporaryFile(delete=False)
        imagefile.write(b'\x00')
        imagefile.close()
        arglist = ['--file', imagefile.name, '--unprotected' if not self.new_image.is_protected else '--protected', '--public' if self.new_image.visibility == 'public' else '--private', '--property', 'Alpha=1', '--property', 'Beta=2', '--tag', self.new_image.tags[0], '--tag', self.new_image.tags[1], self.new_image.name]
        verifylist = [('filename', imagefile.name), ('is_protected', self.new_image.is_protected), ('visibility', self.new_image.visibility), ('properties', {'Alpha': '1', 'Beta': '2'}), ('tags', self.new_image.tags), ('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, is_protected=self.new_image.is_protected, visibility=self.new_image.visibility, Alpha='1', Beta='2', tags=self.new_image.tags, filename=imagefile.name)
        self.assertEqual(self.expected_columns, columns)
        self.assertCountEqual(self.expected_data, data)

    @mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
    def test_image_create__progress_ignore_with_stdin(self, mock_get_data_from_stdin):
        fake_stdin = io.BytesIO(b'some fake data')
        mock_get_data_from_stdin.return_value = fake_stdin
        arglist = ['--progress', self.new_image.name]
        verifylist = [('progress', True), ('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, data=fake_stdin, validate_checksum=False)
        self.assertEqual(self.expected_columns, columns)
        self.assertCountEqual(self.expected_data, data)

    def test_image_create_dead_options(self):
        arglist = ['--store', 'somewhere', self.new_image.name]
        verifylist = [('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch('sys.stdin', side_effect=[None])
    def test_image_create_import(self, raw_input):
        arglist = ['--import', self.new_image.name]
        verifylist = [('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, use_import=True)

    @mock.patch('osc_lib.utils.find_resource')
    @mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
    def test_image_create_from_volume(self, mock_get_data_f, mock_get_vol):
        fake_vol_id = 'fake-volume-id'
        mock_get_data_f.return_value = None

        class FakeVolume:
            id = fake_vol_id
        mock_get_vol.return_value = FakeVolume()
        arglist = ['--volume', fake_vol_id, self.new_image.name]
        verifylist = [('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volumes_mock.upload_to_image.assert_called_with(fake_vol_id, False, self.new_image.name, 'bare', 'raw')

    @mock.patch('osc_lib.utils.find_resource')
    @mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
    def test_image_create_from_volume_fail(self, mock_get_data_f, mock_get_vol):
        fake_vol_id = 'fake-volume-id'
        mock_get_data_f.return_value = None

        class FakeVolume:
            id = fake_vol_id
        mock_get_vol.return_value = FakeVolume()
        arglist = ['--volume', fake_vol_id, self.new_image.name, '--public']
        verifylist = [('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch('osc_lib.utils.find_resource')
    @mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
    def test_image_create_from_volume_v31(self, mock_get_data_f, mock_get_vol):
        self.volume_client.api_version = api_versions.APIVersion('3.1')
        fake_vol_id = 'fake-volume-id'
        mock_get_data_f.return_value = None

        class FakeVolume:
            id = fake_vol_id
        mock_get_vol.return_value = FakeVolume()
        arglist = ['--volume', fake_vol_id, self.new_image.name, '--public']
        verifylist = [('name', self.new_image.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volumes_mock.upload_to_image.assert_called_with(fake_vol_id, False, self.new_image.name, 'bare', 'raw', visibility='public', protected=False)