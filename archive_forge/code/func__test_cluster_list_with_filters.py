import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def _test_cluster_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False, expect=[]):
    clusters_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir, detail=detail)
    self.assertEqual(expect, self.api.calls)
    self.assertThat(clusters_filter, matchers.HasLength(2))