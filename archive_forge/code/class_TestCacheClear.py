from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
class TestCacheClear(fakes.TestImagev2):

    def setUp(self):
        super().setUp()
        self.image_client.clear_cache.return_value = None
        self.cmd = cache.ClearCachedImage(self.app, None)

    def test_cache_clear_no_option(self):
        arglist = []
        verifylist = [('target', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertIsNone(self.image_client.clear_cache.assert_called_with(None))

    def test_cache_clear_queue_option(self):
        arglist = ['--queue']
        verifylist = [('target', 'queue')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.clear_cache.assert_called_once_with('queue')

    def test_cache_clear_cache_option(self):
        arglist = ['--cache']
        verifylist = [('target', 'cache')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.clear_cache.assert_called_once_with('cache')