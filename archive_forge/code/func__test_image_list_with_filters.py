import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def _test_image_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, expect=[]):
    images_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir)
    self.assertEqual(expect, self.api.calls)
    self.assertThat(images_filter, matchers.HasLength(2))