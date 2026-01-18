import uuid
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.tests.unit import base
class TestQoSDSCPMarkingRule(base.TestCase):

    def test_basic(self):
        sot = qos_dscp_marking_rule.QoSDSCPMarkingRule()
        self.assertEqual('dscp_marking_rule', sot.resource_key)
        self.assertEqual('dscp_marking_rules', sot.resources_key)
        self.assertEqual('/qos/policies/%(qos_policy_id)s/dscp_marking_rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = qos_dscp_marking_rule.QoSDSCPMarkingRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['qos_policy_id'], sot.qos_policy_id)
        self.assertEqual(EXAMPLE['dscp_mark'], sot.dscp_mark)