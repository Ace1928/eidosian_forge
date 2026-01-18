from openstack.network.v2 import load_balancer
from openstack.tests.unit import base
class TestLoadBalancer(base.TestCase):

    def test_basic(self):
        sot = load_balancer.LoadBalancer()
        self.assertEqual('loadbalancer', sot.resource_key)
        self.assertEqual('loadbalancers', sot.resources_key)
        self.assertEqual('/lbaas/loadbalancers', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = load_balancer.LoadBalancer(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['listeners'], sot.listener_ids)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['operating_status'], sot.operating_status)
        self.assertEqual(EXAMPLE['provisioning_status'], sot.provisioning_status)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['vip_address'], sot.vip_address)
        self.assertEqual(EXAMPLE['vip_subnet_id'], sot.vip_subnet_id)
        self.assertEqual(EXAMPLE['vip_port_id'], sot.vip_port_id)
        self.assertEqual(EXAMPLE['provider'], sot.provider)
        self.assertEqual(EXAMPLE['pools'], sot.pool_ids)