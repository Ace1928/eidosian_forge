import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import services
def _test_service_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, expect=[]):
    services_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir)
    self.assertEqual(expect, self.api.calls)
    self.assertThat(services_filter, matchers.HasLength(2))