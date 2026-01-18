from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
class TestShowImage(TestImages):

    def setUp(self):
        super(TestShowImage, self).setUp()
        self.image_mock.find_unique.return_value = api_images.Image(None, IMAGE_INFO)
        self.cmd = osc_images.ShowImage(self.app, None)

    def test_image_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_image_show(self):
        arglist = ['image']
        verifylist = [('image', 'image')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_mock.find_unique.assert_called_once_with(name='image')
        expected_columns = ('Description', 'Id', 'Name', 'Status', 'Tags', 'Username')
        self.assertEqual(expected_columns, columns)
        expected_data = ['Image for tests', 'id', 'image', 'Active', '0.1, fake', 'ubuntu']
        self.assertEqual(expected_data, list(data))