from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
class TestRateLimits(base.TestCase):

    def test_basic(self):
        limit_resource = limits.RateLimits()
        self.assertIsNone(limit_resource.resource_key)
        self.assertIsNone(limit_resource.resources_key)
        self.assertEqual('', limit_resource.base_path)
        self.assertFalse(limit_resource.allow_create)
        self.assertFalse(limit_resource.allow_fetch)
        self.assertFalse(limit_resource.allow_delete)
        self.assertFalse(limit_resource.allow_commit)
        self.assertFalse(limit_resource.allow_list)

    def _test_rate_limit(self, expected, actual):
        self.assertEqual(expected[0]['verb'], actual[0].verb)
        self.assertEqual(expected[0]['value'], actual[0].value)
        self.assertEqual(expected[0]['remaining'], actual[0].remaining)
        self.assertEqual(expected[0]['unit'], actual[0].unit)
        self.assertEqual(expected[0]['next-available'], actual[0].next_available)

    def test_make_rate_limits(self):
        limit_resource = limits.RateLimits(**RATE_LIMITS)
        self.assertEqual(RATE_LIMITS['regex'], limit_resource.regex)
        self.assertEqual(RATE_LIMITS['uri'], limit_resource.uri)
        self._test_rate_limit(RATE_LIMITS['limit'], limit_resource.limits)