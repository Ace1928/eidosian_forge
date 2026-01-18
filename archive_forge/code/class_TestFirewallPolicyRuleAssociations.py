import uuid
from openstack.network.v2 import firewall_policy
from openstack.network.v2 import firewall_rule
from openstack.tests.functional import base
class TestFirewallPolicyRuleAssociations(base.BaseFunctionalTest):
    POLICY_NAME = uuid.uuid4().hex
    RULE1_NAME = uuid.uuid4().hex
    RULE2_NAME = uuid.uuid4().hex
    POLICY_ID = None
    RULE1_ID = None
    RULE2_ID = None

    def setUp(self):
        super(TestFirewallPolicyRuleAssociations, self).setUp()
        if not self.user_cloud._has_neutron_extension('fwaas_v2'):
            self.skipTest('fwaas_v2 service not supported by cloud')
        rul1 = self.user_cloud.network.create_firewall_rule(name=self.RULE1_NAME)
        assert isinstance(rul1, firewall_rule.FirewallRule)
        self.assertEqual(self.RULE1_NAME, rul1.name)
        rul2 = self.user_cloud.network.create_firewall_rule(name=self.RULE2_NAME)
        assert isinstance(rul2, firewall_rule.FirewallRule)
        self.assertEqual(self.RULE2_NAME, rul2.name)
        pol = self.user_cloud.network.create_firewall_policy(name=self.POLICY_NAME)
        assert isinstance(pol, firewall_policy.FirewallPolicy)
        self.assertEqual(self.POLICY_NAME, pol.name)
        self.RULE1_ID = rul1.id
        self.RULE2_ID = rul2.id
        self.POLICY_ID = pol.id

    def tearDown(self):
        sot = self.user_cloud.network.delete_firewall_policy(self.POLICY_ID, ignore_missing=False)
        self.assertIs(None, sot)
        sot = self.user_cloud.network.delete_firewall_rule(self.RULE1_ID, ignore_missing=False)
        self.assertIs(None, sot)
        sot = self.user_cloud.network.delete_firewall_rule(self.RULE2_ID, ignore_missing=False)
        self.assertIs(None, sot)
        super(TestFirewallPolicyRuleAssociations, self).tearDown()

    def test_insert_rule_into_policy(self):
        policy = self.user_cloud.network.insert_rule_into_policy(self.POLICY_ID, firewall_rule_id=self.RULE1_ID)
        self.assertIn(self.RULE1_ID, policy['firewall_rules'])
        policy = self.user_cloud.network.insert_rule_into_policy(self.POLICY_ID, firewall_rule_id=self.RULE2_ID, insert_before=self.RULE1_ID)
        self.assertEqual(self.RULE1_ID, policy['firewall_rules'][1])
        self.assertEqual(self.RULE2_ID, policy['firewall_rules'][0])

    def test_remove_rule_from_policy(self):
        policy = self.user_cloud.network.insert_rule_into_policy(self.POLICY_ID, firewall_rule_id=self.RULE1_ID)
        self.assertIn(self.RULE1_ID, policy['firewall_rules'])
        policy = self.user_cloud.network.insert_rule_into_policy(self.POLICY_ID, firewall_rule_id=self.RULE2_ID)
        self.assertIn(self.RULE2_ID, policy['firewall_rules'])
        policy = self.user_cloud.network.remove_rule_from_policy(self.POLICY_ID, firewall_rule_id=self.RULE1_ID)
        self.assertNotIn(self.RULE1_ID, policy['firewall_rules'])
        policy = self.user_cloud.network.remove_rule_from_policy(self.POLICY_ID, firewall_rule_id=self.RULE2_ID)
        self.assertNotIn(self.RULE2_ID, policy['firewall_rules'])