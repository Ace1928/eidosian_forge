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
class TestImageShow(TestImage):
    new_image = image_fakes.create_one_image(attrs={'size': 1000})
    _data = image_fakes.create_one_image()
    columns = ('id', 'name', 'owner', 'protected', 'tags', 'visibility')
    data = (_data.id, _data.name, _data.owner_id, _data.is_protected, format_columns.ListColumn(_data.tags), _data.visibility)

    def setUp(self):
        super().setUp()
        self.image_client.find_image = mock.Mock(return_value=self._data)
        self.cmd = _image.ShowImage(self.app, None)

    def test_image_show(self):
        arglist = ['0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.find_image.assert_called_with('0f41529e-7c12-4de8-be2d-181abb825b3c', ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_image_show_human_readable(self):
        self.image_client.find_image.return_value = self.new_image
        arglist = ['--human-readable', self.new_image.id]
        verifylist = [('human_readable', True), ('image', self.new_image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.find_image.assert_called_with(self.new_image.id, ignore_missing=False)
        size_index = columns.index('size')
        self.assertEqual(data[size_index], '1K')