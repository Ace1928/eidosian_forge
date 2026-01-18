from openstack.network.v2 import router
from openstack.tests.functional import base
class TestDVRRouter(base.BaseFunctionalTest):
    ID = None

    def setUp(self):
        super(TestDVRRouter, self).setUp()
        if not self.operator_cloud:
            self.skipTest('Operator cloud is required for this test')
        if not self.operator_cloud._has_neutron_extension('dvr'):
            self.skipTest('dvr service not supported by cloud')
        self.NAME = self.getUniqueString()
        self.UPDATE_NAME = self.getUniqueString()
        sot = self.operator_cloud.network.create_router(name=self.NAME, distributed=True)
        assert isinstance(sot, router.Router)
        self.assertEqual(self.NAME, sot.name)
        self.ID = sot.id

    def tearDown(self):
        sot = self.operator_cloud.network.delete_router(self.ID, ignore_missing=False)
        self.assertIsNone(sot)
        super(TestDVRRouter, self).tearDown()

    def test_find(self):
        sot = self.operator_cloud.network.find_router(self.NAME)
        self.assertEqual(self.ID, sot.id)

    def test_get(self):
        sot = self.operator_cloud.network.get_router(self.ID)
        self.assertEqual(self.NAME, sot.name)
        self.assertEqual(self.ID, sot.id)
        self.assertTrue(sot.is_distributed)

    def test_list(self):
        names = [o.name for o in self.operator_cloud.network.routers()]
        self.assertIn(self.NAME, names)
        dvr = [o.is_distributed for o in self.operator_cloud.network.routers()]
        self.assertTrue(dvr)

    def test_update(self):
        sot = self.operator_cloud.network.update_router(self.ID, name=self.UPDATE_NAME)
        self.assertEqual(self.UPDATE_NAME, sot.name)