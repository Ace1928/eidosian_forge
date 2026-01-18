from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
class TestCacheList(fakes.TestImagev2):
    _cache = fakes.create_cache()
    columns = ['ID', 'State', 'Last Accessed (UTC)', 'Last Modified (UTC)', 'Size', 'Hits']
    cache_list = cache._format_image_cache(dict(fakes.create_cache()))
    datalist = ((image['image_id'], image['state'], image['last_accessed'], image['last_modified'], image['size'], image['hits']) for image in cache_list)

    def setUp(self):
        super().setUp()
        self.image_client.get_image_cache.return_value = self._cache
        self.cmd = cache.ListCachedImage(self.app, None)

    def test_image_cache_list(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.get_image_cache.assert_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(tuple(self.datalist), tuple(data))