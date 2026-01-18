from openstack.network.v2 import quota
from openstack import resource
from openstack.tests.unit import base
class TestQuotaDefault(base.TestCase):

    def test_basic(self):
        sot = quota.QuotaDefault()
        self.assertEqual('quota', sot.resource_key)
        self.assertEqual('quotas', sot.resources_key)
        self.assertEqual('/quotas/%(project)s/default', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)

    def test_make_it(self):
        sot = quota.QuotaDefault(project='FAKE_PROJECT', **EXAMPLE)
        self.assertEqual(EXAMPLE['floatingip'], sot.floating_ips)
        self.assertEqual(EXAMPLE['network'], sot.networks)
        self.assertEqual(EXAMPLE['port'], sot.ports)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['router'], sot.routers)
        self.assertEqual(EXAMPLE['subnet'], sot.subnets)
        self.assertEqual(EXAMPLE['subnetpool'], sot.subnet_pools)
        self.assertEqual(EXAMPLE['security_group_rule'], sot.security_group_rules)
        self.assertEqual(EXAMPLE['security_group'], sot.security_groups)
        self.assertEqual(EXAMPLE['rbac_policy'], sot.rbac_policies)
        self.assertEqual(EXAMPLE['healthmonitor'], sot.health_monitors)
        self.assertEqual(EXAMPLE['listener'], sot.listeners)
        self.assertEqual(EXAMPLE['loadbalancer'], sot.load_balancers)
        self.assertEqual(EXAMPLE['l7policy'], sot.l7_policies)
        self.assertEqual(EXAMPLE['pool'], sot.pools)
        self.assertEqual(EXAMPLE['check_limit'], sot.check_limit)
        self.assertEqual('FAKE_PROJECT', sot.project)