import uuid
from openstackclient.tests.functional.network.v2 import common
class NetworkQosPolicyTests(common.NetworkTests):
    """Functional tests for QoS policy"""

    def setUp(self):
        super().setUp()
        if not self.is_extension_enabled('qos'):
            self.skipTest('No qos extension present')

    def test_qos_rule_create_delete(self):
        policy_name = uuid.uuid4().hex
        self.openstack('network qos policy create ' + policy_name)
        raw_output = self.openstack('network qos policy delete ' + policy_name)
        self.assertEqual('', raw_output)

    def test_qos_policy_list(self):
        policy_name = uuid.uuid4().hex
        json_output = self.openstack('network qos policy create ' + policy_name, parse_output=True)
        self.addCleanup(self.openstack, 'network qos policy delete ' + policy_name)
        self.assertEqual(policy_name, json_output['name'])
        json_output = self.openstack('network qos policy list', parse_output=True)
        self.assertIn(policy_name, [p['Name'] for p in json_output])

    def test_qos_policy_set(self):
        policy_name = uuid.uuid4().hex
        json_output = self.openstack('network qos policy create ' + policy_name, parse_output=True)
        self.addCleanup(self.openstack, 'network qos policy delete ' + policy_name)
        self.assertEqual(policy_name, json_output['name'])
        self.openstack('network qos policy set ' + '--share ' + policy_name)
        json_output = self.openstack('network qos policy show ' + policy_name, parse_output=True)
        self.assertTrue(json_output['shared'])
        self.openstack('network qos policy set ' + '--no-share ' + '--no-default ' + policy_name)
        json_output = self.openstack('network qos policy show ' + policy_name, parse_output=True)
        self.assertFalse(json_output['shared'])
        self.assertFalse(json_output['is_default'])