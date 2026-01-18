import fixtures
import http.client as http
from oslo_utils import units
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional.v2.test_images import get_enforcer_class
from glance.tests import utils as test_utils
def _assert_usage(self, expected):
    usage = self.api_get('/v2/info/usage')
    usage = usage.json['usage']
    for item in ('count', 'size', 'stage'):
        key = 'image_%s_total' % item
        self.assertEqual(expected[key], usage[key], 'Mismatch in %s' % key)
    self.assertEqual(expected['image_count_uploading'], usage['image_count_uploading'])