from openstack.identity.v3 import trust
from openstack.tests.unit import base
class TestTrust(base.TestCase):

    def test_basic(self):
        sot = trust.Trust()
        self.assertEqual('trust', sot.resource_key)
        self.assertEqual('trusts', sot.resources_key)
        self.assertEqual('/OS-TRUST/trusts', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = trust.Trust(**EXAMPLE)
        self.assertEqual(EXAMPLE['allow_redelegation'], sot.allow_redelegation)
        self.assertEqual(EXAMPLE['expires_at'], sot.expires_at)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertTrue(sot.is_impersonation)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['role_links'], sot.role_links)
        self.assertEqual(EXAMPLE['redelegated_trust_id'], sot.redelegated_trust_id)
        self.assertEqual(EXAMPLE['remaining_uses'], sot.remaining_uses)
        self.assertEqual(EXAMPLE['trustee_user_id'], sot.trustee_user_id)
        self.assertEqual(EXAMPLE['trustor_user_id'], sot.trustor_user_id)
        self.assertEqual(EXAMPLE['roles'], sot.roles)
        self.assertEqual(EXAMPLE['redelegation_count'], sot.redelegation_count)