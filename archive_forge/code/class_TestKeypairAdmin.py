from openstack.compute.v2 import keypair
from openstack.tests.functional import base
class TestKeypairAdmin(base.BaseFunctionalTest):

    def setUp(self):
        super(TestKeypairAdmin, self).setUp()
        self._set_operator_cloud(interface='admin')
        self.NAME = self.getUniqueString().split('.')[-1]
        self.USER = self.operator_cloud.list_users()[0]
        sot = self.conn.compute.create_keypair(name=self.NAME, user_id=self.USER.id)
        assert isinstance(sot, keypair.Keypair)
        self.assertEqual(self.NAME, sot.name)
        self.assertEqual(self.USER.id, sot.user_id)
        self._keypair = sot

    def tearDown(self):
        sot = self.conn.compute.delete_keypair(self._keypair)
        self.assertIsNone(sot)
        super(TestKeypairAdmin, self).tearDown()

    def test_get(self):
        sot = self.conn.compute.get_keypair(self.NAME)
        self.assertEqual(self.NAME, sot.name)
        self.assertEqual(self.NAME, sot.id)
        self.assertEqual(self.USER.id, sot.user_id)