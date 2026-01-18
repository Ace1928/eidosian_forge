from openstack.tests.functional.shared_file_system import base
class StoragePoolTest(base.BaseSharedFileSystemTest):

    def test_storage_pools(self):
        pools = self.operator_cloud.shared_file_system.storage_pools()
        self.assertGreater(len(list(pools)), 0)
        for pool in pools:
            for attribute in ('pool', 'name', 'host', 'backend', 'capabilities'):
                self.assertTrue(hasattr(pool, attribute))