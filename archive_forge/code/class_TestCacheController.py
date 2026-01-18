import testtools
from unittest import mock
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests import utils
from glanceclient.v2 import cache
class TestCacheController(testtools.TestCase):

    def setUp(self):
        super(TestCacheController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.controller = cache.Controller(self.api)

    @mock.patch.object(common_utils, 'has_version')
    def test_list_cached(self, mock_has_version):
        mock_has_version.return_value = True
        images = self.controller.list()
        self.assertEqual(2, len(images['cached_images']))
        self.assertEqual(2, len(images['queued_images']))

    @mock.patch.object(common_utils, 'has_version')
    def test_list_cached_empty_response(self, mock_has_version):
        dummy_fixtures = {'/v2/cache': {'GET': ({}, {'cached_images': [], 'queued_images': []})}}
        dummy_api = utils.FakeAPI(dummy_fixtures)
        dummy_controller = cache.Controller(dummy_api)
        mock_has_version.return_value = True
        images = dummy_controller.list()
        self.assertEqual(0, len(images['cached_images']))
        self.assertEqual(0, len(images['queued_images']))

    @mock.patch.object(common_utils, 'has_version')
    def test_queue_image(self, mock_has_version):
        mock_has_version.return_value = True
        image_id = '3a4560a1-e585-443e-9b39-553b46ec92d1'
        self.controller.queue(image_id)
        expect = [('PUT', '/v2/cache/%s' % image_id, {}, None)]
        self.assertEqual(expect, self.api.calls)

    @mock.patch.object(common_utils, 'has_version')
    def test_cache_clear_with_header(self, mock_has_version):
        mock_has_version.return_value = True
        self.controller.clear('cache')
        expect = [('DELETE', '/v2/cache', {'x-image-cache-clear-target': 'cache'}, None)]
        self.assertEqual(expect, self.api.calls)

    @mock.patch.object(common_utils, 'has_version')
    def test_cache_delete(self, mock_has_version):
        mock_has_version.return_value = True
        image_id = '3a4560a1-e585-443e-9b39-553b46ec92d1'
        self.controller.delete(image_id)
        expect = [('DELETE', '/v2/cache/%s' % image_id, {}, None)]
        self.assertEqual(expect, self.api.calls)

    @mock.patch.object(common_utils, 'has_version')
    def test_cache_not_supported(self, mock_has_version):
        mock_has_version.return_value = False
        self.assertRaises(exc.HTTPNotImplemented, self.controller.list)