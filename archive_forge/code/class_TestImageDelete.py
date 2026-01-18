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
class TestImageDelete(TestImage):

    def setUp(self):
        super().setUp()
        self.image_client.delete_image.return_value = None
        self.cmd = _image.DeleteImage(self.app, None)

    def test_image_delete_no_options(self):
        images = image_fakes.create_images(count=1)
        arglist = [images[0].id]
        verifylist = [('images', [images[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        result = self.cmd.take_action(parsed_args)
        self.image_client.delete_image.assert_called_with(images[0].id, store=parsed_args.store, ignore_missing=False)
        self.assertIsNone(result)

    def test_image_delete_from_store(self):
        images = image_fakes.create_images(count=1)
        arglist = [images[0].id, '--store', 'store1']
        verifylist = [('images', [images[0].id]), ('store', 'store1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        result = self.cmd.take_action(parsed_args)
        self.image_client.delete_image.assert_called_with(images[0].id, store=parsed_args.store, ignore_missing=False)
        self.assertIsNone(result)

    def test_image_delete_multi_images(self):
        images = image_fakes.create_images(count=3)
        arglist = [i.id for i in images]
        verifylist = [('images', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        result = self.cmd.take_action(parsed_args)
        calls = [mock.call(i.id, store=parsed_args.store, ignore_missing=False) for i in images]
        self.image_client.delete_image.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_image_delete_from_store_without_multi_backend(self):
        images = image_fakes.create_images(count=1)
        arglist = [images[0].id, '--store', 'store1']
        verifylist = [('images', [images[0].id]), ('store', 'store1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.image_client.find_image.side_effect = images
        self.image_client.delete_image.side_effect = sdk_exceptions.ResourceNotFound
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('Multi Backend support not enabled', str(exc))

    def test_image_delete_multi_images_exception(self):
        images = image_fakes.create_images(count=2)
        arglist = [images[0].id, images[1].id, 'x-y-x']
        verifylist = [('images', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret_find = [images[0], images[1], sdk_exceptions.ResourceNotFound()]
        self.image_client.find_image.side_effect = ret_find
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        calls = [mock.call(i.id, store=parsed_args.store, ignore_missing=False) for i in images]
        self.image_client.delete_image.assert_has_calls(calls)