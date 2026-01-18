from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
class TestRemoveImageTags(TestImages):

    def setUp(self):
        super(TestRemoveImageTags, self).setUp()
        self.image_mock.update_tags.return_value = api_images.Image(None, IMAGE_INFO)
        self.image_mock.find_unique.return_value = api_images.Image(None, IMAGE_INFO)
        self.cmd = osc_images.RemoveImageTags(self.app, None)

    def test_image_tags_remove_both_options(self):
        arglist = ['id', '--all', '--tags', 'fake']
        verifylist = [('image', 'id'), ('all', True), ('tags', ['fake'])]
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_image_tags_remove(self):
        arglist = ['image', '--tags', 'fake']
        verifylist = [('image', 'image'), ('tags', ['fake'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_mock.find_unique.assert_called_with(name='image')
        self.image_mock.update_tags.assert_called_once_with('id', ['0.1'])

    def test_image_tags_remove_all(self):
        arglist = ['image', '--all']
        verifylist = [('image', 'image'), ('all', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_mock.find_unique.assert_called_with(name='image')
        self.image_mock.update_tags.assert_called_once_with('id', [])