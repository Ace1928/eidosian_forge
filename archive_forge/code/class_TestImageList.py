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
class TestImageList(TestImage):
    _image = image_fakes.create_one_image()
    columns = ('ID', 'Name', 'Status')
    datalist = ((_image.id, _image.name, None),)

    def setUp(self):
        super().setUp()
        self.image_client.images.side_effect = [[self._image], []]
        self.cmd = _image.ListImage(self.app, None)

    def test_image_list_no_options(self):
        arglist = []
        verifylist = [('visibility', None), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_public_option(self):
        arglist = ['--public']
        verifylist = [('visibility', 'public'), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='public')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_private_option(self):
        arglist = ['--private']
        verifylist = [('visibility', 'private'), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='private')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_community_option(self):
        arglist = ['--community']
        verifylist = [('visibility', 'community'), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='community')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_image_list_shared_option(self):
        arglist = ['--shared']
        verifylist = [('visibility', 'shared'), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='shared')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_all_option(self):
        arglist = ['--all']
        verifylist = [('visibility', 'all'), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='all')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_shared_member_status_option(self):
        arglist = ['--shared', '--member-status', 'all']
        verifylist = [('visibility', 'shared'), ('long', False), ('member_status', 'all')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(visibility='shared', member_status='all')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_image_list_shared_member_status_lower(self):
        arglist = ['--shared', '--member-status', 'ALl']
        verifylist = [('visibility', 'shared'), ('long', False), ('member_status', 'all')]
        self.check_parser(self.cmd, arglist, verifylist)

    def test_image_list_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with()
        collist = ('ID', 'Name', 'Disk Format', 'Container Format', 'Size', 'Checksum', 'Status', 'Visibility', 'Protected', 'Project', 'Tags')
        self.assertEqual(collist, columns)
        datalist = ((self._image.id, self._image.name, None, None, None, None, None, self._image.visibility, self._image.is_protected, self._image.owner_id, format_columns.ListColumn(self._image.tags)),)
        self.assertCountEqual(datalist, tuple(data))

    @mock.patch('osc_lib.api.utils.simple_filter')
    def test_image_list_property_option(self, sf_mock):
        sf_mock.return_value = [copy.deepcopy(self._image)]
        arglist = ['--property', 'a=1']
        verifylist = [('property', {'a': '1'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with()
        sf_mock.assert_called_with([self._image], attr='a', value='1', property_field='properties')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    @mock.patch('osc_lib.utils.sort_items')
    def test_image_list_sort_option(self, si_mock):
        si_mock.return_value = [copy.deepcopy(self._image)]
        arglist = ['--sort', 'name:asc']
        verifylist = [('sort', 'name:asc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with()
        si_mock.assert_called_with([self._image], 'name:asc', str)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    def test_image_list_limit_option(self):
        ret_limit = 1
        arglist = ['--limit', str(ret_limit)]
        verifylist = [('limit', ret_limit)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(limit=ret_limit, paginated=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(ret_limit, len(tuple(data)))

    def test_image_list_project_option(self):
        self.image_client.find_image = mock.Mock(return_value=self._image)
        arglist = ['--project', 'nova']
        verifylist = [('project', 'nova')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, tuple(data))

    @mock.patch('osc_lib.utils.find_resource')
    def test_image_list_marker_option(self, fr_mock):
        self.image_client.find_image = mock.Mock(return_value=self._image)
        arglist = ['--marker', 'graven']
        verifylist = [('marker', 'graven')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(marker=self._image.id)
        self.image_client.find_image.assert_called_with('graven', ignore_missing=False)

    def test_image_list_name_option(self):
        arglist = ['--name', 'abc']
        verifylist = [('name', 'abc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(name='abc')

    def test_image_list_status_option(self):
        arglist = ['--status', 'active']
        verifylist = [('status', 'active')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(status='active')

    def test_image_list_hidden_option(self):
        arglist = ['--hidden']
        verifylist = [('is_hidden', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(is_hidden=True)

    def test_image_list_tag_option(self):
        arglist = ['--tag', 'abc', '--tag', 'cba']
        verifylist = [('tag', ['abc', 'cba'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.images.assert_called_with(tag=['abc', 'cba'])