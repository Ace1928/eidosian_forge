from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
class TestVolumeSet(TestVolume):
    volume_type = volume_fakes.create_one_volume_type()

    def setUp(self):
        super().setUp()
        self.new_volume = volume_fakes.create_one_volume()
        self.volumes_mock.get.return_value = self.new_volume
        self.types_mock.get.return_value = self.volume_type
        self.cmd = volume.SetVolume(self.app, None)

    def test_volume_set_property(self):
        arglist = ['--property', 'a=b', '--property', 'c=d', self.new_volume.id]
        verifylist = [('property', {'a': 'b', 'c': 'd'}), ('volume', self.new_volume.id), ('bootable', False), ('non_bootable', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.volumes_mock.set_metadata.assert_called_with(self.new_volume.id, parsed_args.property)

    def test_volume_set_image_property(self):
        arglist = ['--image-property', 'Alpha=a', '--image-property', 'Beta=b', self.new_volume.id]
        verifylist = [('image_property', {'Alpha': 'a', 'Beta': 'b'}), ('volume', self.new_volume.id), ('bootable', False), ('non_bootable', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.volumes_mock.set_image_metadata.assert_called_with(self.new_volume.id, parsed_args.image_property)

    def test_volume_set_state(self):
        arglist = ['--state', 'error', self.new_volume.id]
        verifylist = [('read_only', False), ('read_write', False), ('state', 'error'), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.reset_state.assert_called_with(self.new_volume.id, 'error')
        self.volumes_mock.update_readonly_flag.assert_not_called()
        self.assertIsNone(result)

    def test_volume_set_state_failed(self):
        self.volumes_mock.reset_state.side_effect = exceptions.CommandError()
        arglist = ['--state', 'error', self.new_volume.id]
        verifylist = [('state', 'error'), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('One or more of the set operations failed', str(e))
        self.volumes_mock.reset_state.assert_called_with(self.new_volume.id, 'error')

    def test_volume_set_attached(self):
        arglist = ['--attached', self.new_volume.id]
        verifylist = [('attached', True), ('detached', False), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.reset_state.assert_called_with(self.new_volume.id, attach_status='attached', state=None)
        self.assertIsNone(result)

    def test_volume_set_detached(self):
        arglist = ['--detached', self.new_volume.id]
        verifylist = [('attached', False), ('detached', True), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.reset_state.assert_called_with(self.new_volume.id, attach_status='detached', state=None)
        self.assertIsNone(result)

    def test_volume_set_bootable(self):
        arglist = [['--bootable', self.new_volume.id], ['--non-bootable', self.new_volume.id]]
        verifylist = [[('bootable', True), ('non_bootable', False), ('volume', self.new_volume.id)], [('bootable', False), ('non_bootable', True), ('volume', self.new_volume.id)]]
        for index in range(len(arglist)):
            parsed_args = self.check_parser(self.cmd, arglist[index], verifylist[index])
            self.cmd.take_action(parsed_args)
            self.volumes_mock.set_bootable.assert_called_with(self.new_volume.id, verifylist[index][0][1])

    def test_volume_set_readonly(self):
        arglist = ['--read-only', self.new_volume.id]
        verifylist = [('read_only', True), ('read_write', False), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.update_readonly_flag.assert_called_once_with(self.new_volume.id, True)
        self.assertIsNone(result)

    def test_volume_set_read_write(self):
        arglist = ['--read-write', self.new_volume.id]
        verifylist = [('read_only', False), ('read_write', True), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.update_readonly_flag.assert_called_once_with(self.new_volume.id, False)
        self.assertIsNone(result)

    def test_volume_set_type(self):
        arglist = ['--type', self.volume_type.id, self.new_volume.id]
        verifylist = [('retype_policy', None), ('type', self.volume_type.id), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.retype.assert_called_once_with(self.new_volume.id, self.volume_type.id, 'never')
        self.assertIsNone(result)

    def test_volume_set_type_with_policy(self):
        arglist = ['--retype-policy', 'on-demand', '--type', self.volume_type.id, self.new_volume.id]
        verifylist = [('retype_policy', 'on-demand'), ('type', self.volume_type.id), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.retype.assert_called_once_with(self.new_volume.id, self.volume_type.id, 'on-demand')
        self.assertIsNone(result)

    @mock.patch.object(volume.LOG, 'warning')
    def test_volume_set_with_only_retype_policy(self, mock_warning):
        arglist = ['--retype-policy', 'on-demand', self.new_volume.id]
        verifylist = [('retype_policy', 'on-demand'), ('volume', self.new_volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volumes_mock.retype.assert_not_called()
        mock_warning.assert_called_with("'--retype-policy' option will not work without '--type' option")
        self.assertIsNone(result)