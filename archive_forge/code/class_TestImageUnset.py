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
class TestImageUnset(TestImage):

    def setUp(self):
        super().setUp()
        attrs = {}
        attrs['tags'] = ['test']
        attrs['hw_rng_model'] = 'virtio'
        attrs['prop'] = 'test'
        attrs['prop2'] = 'fake'
        self.image = image_fakes.create_one_image(attrs)
        self.image_client.find_image.return_value = self.image
        self.image_client.remove_tag.return_value = self.image
        self.image_client.update_image.return_value = self.image
        self.cmd = _image.UnsetImage(self.app, None)

    def test_image_unset_no_options(self):
        arglist = ['0f41529e-7c12-4de8-be2d-181abb825b3c']
        verifylist = [('image', '0f41529e-7c12-4de8-be2d-181abb825b3c')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)

    def test_image_unset_tag_option(self):
        arglist = ['--tag', 'test', self.image.id]
        verifylist = [('tags', ['test']), ('image', self.image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.image_client.remove_tag.assert_called_with(self.image.id, 'test')
        self.assertIsNone(result)

    def test_image_unset_property_option(self):
        arglist = ['--property', 'hw_rng_model', '--property', 'prop', self.image.id]
        verifylist = [('properties', ['hw_rng_model', 'prop']), ('image', self.image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.image_client.update_image.assert_called_with(self.image, properties={'prop2': 'fake'})
        self.assertIsNone(result)

    def test_image_unset_mixed_option(self):
        arglist = ['--tag', 'test', '--property', 'hw_rng_model', '--property', 'prop', self.image.id]
        verifylist = [('tags', ['test']), ('properties', ['hw_rng_model', 'prop']), ('image', self.image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.image_client.update_image.assert_called_with(self.image, properties={'prop2': 'fake'})
        self.image_client.remove_tag.assert_called_with(self.image.id, 'test')
        self.assertIsNone(result)