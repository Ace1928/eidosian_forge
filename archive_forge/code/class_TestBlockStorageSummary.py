import copy
from openstack.block_storage.v3 import block_storage_summary as summary
from openstack.tests.unit import base
class TestBlockStorageSummary(base.TestCase):

    def test_basic(self):
        summary_resource = summary.BlockStorageSummary()
        self.assertEqual(None, summary_resource.resource_key)
        self.assertEqual(None, summary_resource.resources_key)
        self.assertEqual('/volumes/summary', summary_resource.base_path)
        self.assertTrue(summary_resource.allow_fetch)
        self.assertFalse(summary_resource.allow_create)
        self.assertFalse(summary_resource.allow_commit)
        self.assertFalse(summary_resource.allow_delete)
        self.assertFalse(summary_resource.allow_list)

    def test_get_summary_312(self):
        summary_resource = summary.BlockStorageSummary(**BLOCK_STORAGE_SUMMARY_312)
        self.assertEqual(BLOCK_STORAGE_SUMMARY_312['total_size'], summary_resource.total_size)
        self.assertEqual(BLOCK_STORAGE_SUMMARY_312['total_count'], summary_resource.total_count)

    def test_get_summary_326(self):
        summary_resource = summary.BlockStorageSummary(**BLOCK_STORAGE_SUMMARY_326)
        self.assertEqual(BLOCK_STORAGE_SUMMARY_326['total_size'], summary_resource.total_size)
        self.assertEqual(BLOCK_STORAGE_SUMMARY_326['total_count'], summary_resource.total_count)
        self.assertEqual(BLOCK_STORAGE_SUMMARY_326['metadata'], summary_resource.metadata)