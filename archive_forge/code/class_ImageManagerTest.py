import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
class ImageManagerTest(testtools.TestCase):

    def setUp(self):
        super(ImageManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = images.ImageManager(self.api)

    def test_image_list(self):
        images = self.mgr.list()
        expect = [('GET', '/v1/images/', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(images, matchers.HasLength(2))

    def _test_image_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, expect=[]):
        images_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir)
        self.assertEqual(expect, self.api.calls)
        self.assertThat(images_filter, matchers.HasLength(2))

    def test_image_list_with_limit(self):
        expect = [('GET', '/v1/images/?limit=2', {}, None)]
        self._test_image_list_with_filters(limit=2, expect=expect)

    def test_image_list_with_marker(self):
        expect = [('GET', '/v1/images/?marker=%s' % IMAGE2['image_id'], {}, None)]
        self._test_image_list_with_filters(marker=IMAGE2['image_id'], expect=expect)

    def test_image_list_with_marker_limit(self):
        expect = [('GET', '/v1/images/?limit=2&marker=%s' % IMAGE2['image_id'], {}, None)]
        self._test_image_list_with_filters(limit=2, marker=IMAGE2['image_id'], expect=expect)

    def test_image_list_with_sort_dir(self):
        expect = [('GET', '/v1/images/?sort_dir=asc', {}, None)]
        self._test_image_list_with_filters(sort_dir='asc', expect=expect)

    def test_image_list_with_sort_key(self):
        expect = [('GET', '/v1/images/?sort_key=image_id', {}, None)]
        self._test_image_list_with_filters(sort_key='image_id', expect=expect)

    def test_image_list_with_sort_key_dir(self):
        expect = [('GET', '/v1/images/?sort_key=image_id&sort_dir=desc', {}, None)]
        self._test_image_list_with_filters(sort_key='image_id', sort_dir='desc', expect=expect)

    def test_image_search(self):
        images = self.mgr.search_image(**SEARCH_IMAGE)
        url = '/v1/images/%s/search?image_driver=%s' % (IMAGE3['image'], IMAGE3['image_driver'])
        expect = [('GET', url, {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(images, matchers.HasLength(1))