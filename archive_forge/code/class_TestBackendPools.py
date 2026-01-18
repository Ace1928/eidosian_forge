from openstack.block_storage.v2 import stats
from openstack.tests.unit import base
class TestBackendPools(base.TestCase):

    def setUp(self):
        super(TestBackendPools, self).setUp()

    def test_basic(self):
        sot = stats.Pools(POOLS)
        self.assertEqual('', sot.resource_key)
        self.assertEqual('pools', sot.resources_key)
        self.assertEqual('/scheduler-stats/get_pools?detail=True', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_commit)