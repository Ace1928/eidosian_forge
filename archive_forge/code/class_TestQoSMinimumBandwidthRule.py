import uuid
from openstack.network.v2 import qos_minimum_bandwidth_rule
from openstack.tests.unit import base
class TestQoSMinimumBandwidthRule(base.TestCase):

    def test_basic(self):
        sot = qos_minimum_bandwidth_rule.QoSMinimumBandwidthRule()
        self.assertEqual('minimum_bandwidth_rule', sot.resource_key)
        self.assertEqual('minimum_bandwidth_rules', sot.resources_key)
        self.assertEqual('/qos/policies/%(qos_policy_id)s/minimum_bandwidth_rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = qos_minimum_bandwidth_rule.QoSMinimumBandwidthRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['qos_policy_id'], sot.qos_policy_id)
        self.assertEqual(EXAMPLE['min_kbps'], sot.min_kbps)
        self.assertEqual(EXAMPLE['direction'], sot.direction)