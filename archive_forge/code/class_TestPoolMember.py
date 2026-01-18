from openstack.network.v2 import pool_member
from openstack.tests.unit import base
class TestPoolMember(base.TestCase):

    def test_basic(self):
        sot = pool_member.PoolMember()
        self.assertEqual('member', sot.resource_key)
        self.assertEqual('members', sot.resources_key)
        self.assertEqual('/lbaas/pools/%(pool_id)s/members', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = pool_member.PoolMember(**EXAMPLE)
        self.assertEqual(EXAMPLE['address'], sot.address)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['protocol_port'], sot.protocol_port)
        self.assertEqual(EXAMPLE['subnet_id'], sot.subnet_id)
        self.assertEqual(EXAMPLE['weight'], sot.weight)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['pool_id'], sot.pool_id)