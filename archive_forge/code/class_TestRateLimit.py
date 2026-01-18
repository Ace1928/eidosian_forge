from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
class TestRateLimit(base.TestCase):

    def test_basic(self):
        limit_resource = limits.RateLimit()
        self.assertIsNone(limit_resource.resource_key)
        self.assertIsNone(limit_resource.resources_key)
        self.assertEqual('', limit_resource.base_path)
        self.assertFalse(limit_resource.allow_create)
        self.assertFalse(limit_resource.allow_fetch)
        self.assertFalse(limit_resource.allow_delete)
        self.assertFalse(limit_resource.allow_commit)
        self.assertFalse(limit_resource.allow_list)

    def test_make_rate_limit(self):
        limit_resource = limits.RateLimit(**RATE_LIMIT)
        self.assertEqual(RATE_LIMIT['verb'], limit_resource.verb)
        self.assertEqual(RATE_LIMIT['value'], limit_resource.value)
        self.assertEqual(RATE_LIMIT['remaining'], limit_resource.remaining)
        self.assertEqual(RATE_LIMIT['unit'], limit_resource.unit)
        self.assertEqual(RATE_LIMIT['next-available'], limit_resource.next_available)