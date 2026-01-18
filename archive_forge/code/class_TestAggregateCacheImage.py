from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestAggregateCacheImage(TestAggregate):
    images = image_fakes.create_images(count=2)

    def setUp(self):
        super(TestAggregateCacheImage, self).setUp()
        self.compute_sdk_client.find_aggregate.return_value = self.fake_ag
        self.find_image_mock = mock.Mock(side_effect=self.images)
        self.app.client_manager.sdk_connection.image.find_image = self.find_image_mock
        self.cmd = aggregate.CacheImageForAggregate(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_aggregate_not_supported(self, sm_mock):
        arglist = ['ag1', 'im1']
        verifylist = [('aggregate', 'ag1'), ('image', ['im1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_aggregate_add_single_image(self, sm_mock):
        arglist = ['ag1', 'im1']
        verifylist = [('aggregate', 'ag1'), ('image', ['im1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_aggregate.assert_called_once_with(parsed_args.aggregate, ignore_missing=False)
        self.compute_sdk_client.aggregate_precache_images.assert_called_once_with(self.fake_ag.id, [self.images[0].id])

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_aggregate_add_multiple_images(self, sm_mock):
        arglist = ['ag1', 'im1', 'im2']
        verifylist = [('aggregate', 'ag1'), ('image', ['im1', 'im2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_aggregate.assert_called_once_with(parsed_args.aggregate, ignore_missing=False)
        self.compute_sdk_client.aggregate_precache_images.assert_called_once_with(self.fake_ag.id, [self.images[0].id, self.images[1].id])