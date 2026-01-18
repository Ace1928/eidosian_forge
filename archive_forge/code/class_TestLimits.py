import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import limits
from openstack.tests.unit import base
class TestLimits(base.TestCase):

    def test_basic(self):
        sot = limits.Limits()
        self.assertEqual('limits', sot.resource_key)
        self.assertEqual('/limits', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'tenant_id': 'tenant_id'}, sot._query_mapping._mapping)

    def test_get(self):
        sess = mock.Mock(spec=adapter.Adapter)
        sess.default_microversion = None
        resp = mock.Mock()
        sess.get.return_value = resp
        resp.json.return_value = copy.deepcopy(LIMITS_BODY)
        resp.headers = {}
        resp.status_code = 200
        sot = limits.Limits().fetch(sess)
        self.assertEqual(ABSOLUTE_LIMITS['maxImageMeta'], sot.absolute.image_meta)
        self.assertEqual(ABSOLUTE_LIMITS['maxSecurityGroupRules'], sot.absolute.security_group_rules)
        self.assertEqual(ABSOLUTE_LIMITS['maxSecurityGroups'], sot.absolute.security_groups)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerMeta'], sot.absolute.server_meta)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalCores'], sot.absolute.total_cores)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalFloatingIps'], sot.absolute.floating_ips)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalInstances'], sot.absolute.instances)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalKeypairs'], sot.absolute.keypairs)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalRAMSize'], sot.absolute.total_ram)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerGroups'], sot.absolute.server_groups)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerGroupMembers'], sot.absolute.server_group_members)
        self.assertEqual(ABSOLUTE_LIMITS['totalFloatingIpsUsed'], sot.absolute.floating_ips_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalSecurityGroupsUsed'], sot.absolute.security_groups_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalRAMUsed'], sot.absolute.total_ram_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalInstancesUsed'], sot.absolute.instances_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalServerGroupsUsed'], sot.absolute.server_groups_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalCoresUsed'], sot.absolute.total_cores_used)
        self.assertEqual(RATE_LIMIT['uri'], sot.rate[0].uri)
        self.assertEqual(RATE_LIMIT['regex'], sot.rate[0].regex)
        self.assertEqual(RATE_LIMIT['limit'], sot.rate[0].limits)
        dsot = sot.to_dict()
        self.assertIsInstance(dsot['rate'][0], dict)
        self.assertIsInstance(dsot['absolute'], dict)
        self.assertEqual(RATE_LIMIT['uri'], dsot['rate'][0]['uri'])
        self.assertEqual(ABSOLUTE_LIMITS['totalSecurityGroupsUsed'], dsot['absolute']['security_groups_used'])