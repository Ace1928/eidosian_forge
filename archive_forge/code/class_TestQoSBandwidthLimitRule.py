import uuid
from openstack.network.v2 import qos_bandwidth_limit_rule
from openstack.tests.unit import base
class TestQoSBandwidthLimitRule(base.TestCase):

    def test_basic(self):
        sot = qos_bandwidth_limit_rule.QoSBandwidthLimitRule()
        self.assertEqual('bandwidth_limit_rule', sot.resource_key)
        self.assertEqual('bandwidth_limit_rules', sot.resources_key)
        self.assertEqual('/qos/policies/%(qos_policy_id)s/bandwidth_limit_rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = qos_bandwidth_limit_rule.QoSBandwidthLimitRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['qos_policy_id'], sot.qos_policy_id)
        self.assertEqual(EXAMPLE['max_kbps'], sot.max_kbps)
        self.assertEqual(EXAMPLE['max_burst_kbps'], sot.max_burst_kbps)
        self.assertEqual(EXAMPLE['direction'], sot.direction)