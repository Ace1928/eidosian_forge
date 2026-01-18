import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def _test_coe_service_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False, expect=[]):
    mservices_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir, detail=detail)
    self.assertEqual(expect, self.api.calls)
    self.assertThat(mservices_filter, matchers.HasLength(2))