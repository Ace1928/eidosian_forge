from openstack.network.v2 import vpn_ike_policy
from openstack.tests.unit import base
class TestVpnIkePolicy(base.TestCase):

    def test_basic(self):
        sot = vpn_ike_policy.VpnIkePolicy()
        self.assertEqual('ikepolicy', sot.resource_key)
        self.assertEqual('ikepolicies', sot.resources_key)
        self.assertEqual('/vpn/ikepolicies', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = vpn_ike_policy.VpnIkePolicy(**EXAMPLE)
        self.assertEqual(EXAMPLE['auth_algorithm'], sot.auth_algorithm)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['encryption_algorithm'], sot.encryption_algorithm)
        self.assertEqual(EXAMPLE['ike_version'], sot.ike_version)
        self.assertEqual(EXAMPLE['lifetime'], sot.lifetime)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['pfs'], sot.pfs)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['phase1_negotiation_mode'], sot.phase1_negotiation_mode)
        self.assertEqual(EXAMPLE['units'], sot.units)
        self.assertEqual(EXAMPLE['value'], sot.value)