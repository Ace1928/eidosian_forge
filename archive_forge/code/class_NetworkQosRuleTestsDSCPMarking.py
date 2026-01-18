import uuid
from openstackclient.tests.functional.network.v2 import common
class NetworkQosRuleTestsDSCPMarking(NetworkQosTests):
    """Functional tests for QoS DSCP marking rule"""

    def setUp(self):
        super().setUp()
        self.QOS_POLICY_NAME = 'qos_policy_%s' % uuid.uuid4().hex
        self.openstack('network qos policy create %s' % self.QOS_POLICY_NAME)
        self.addCleanup(self.openstack, 'network qos policy delete %s' % self.QOS_POLICY_NAME)
        cmd_output = self.openstack('network qos rule create --type dscp-marking --dscp-mark 8 %s' % self.QOS_POLICY_NAME, parse_output=True)
        self.RULE_ID = cmd_output['id']
        self.addCleanup(self.openstack, 'network qos rule delete %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID))
        self.assertTrue(self.RULE_ID)

    def test_qos_rule_create_delete(self):
        policy_name = uuid.uuid4().hex
        self.openstack('network qos policy create %s' % policy_name)
        self.addCleanup(self.openstack, 'network qos policy delete %s' % policy_name)
        rule = self.openstack('network qos rule create --type dscp-marking --dscp-mark 8 %s' % policy_name, parse_output=True)
        raw_output = self.openstack('network qos rule delete %s %s' % (policy_name, rule['id']))
        self.assertEqual('', raw_output)

    def test_qos_rule_list(self):
        cmd_output = self.openstack('network qos rule list %s' % self.QOS_POLICY_NAME, parse_output=True)
        self.assertIn(self.RULE_ID, [rule['ID'] for rule in cmd_output])

    def test_qos_rule_show(self):
        cmd_output = self.openstack('network qos rule show %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID), parse_output=True)
        self.assertEqual(self.RULE_ID, cmd_output['id'])

    def test_qos_rule_set(self):
        self.openstack('network qos rule set --dscp-mark 32 %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID))
        cmd_output = self.openstack('network qos rule show %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID), parse_output=True)
        self.assertEqual(32, cmd_output['dscp_mark'])