from openstack.tests.functional.block_storage.v3 import base
class ResourceFilters(base.BaseBlockStorageTest):

    def test_get(self):
        resource_filters = list(self.conn.block_storage.resource_filters())
        for rf in resource_filters:
            self.assertIsInstance(rf.filters, list)
            self.assertIsInstance(rf.resource, str)