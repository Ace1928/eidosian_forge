import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
@ddt.ddt
class ShareAccessAllowTestCase(base.OSCClientTestBase):

    def test_share_access_allow(self):
        share = self.create_share()
        access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
        self.assertEqual(access_rule['share_id'], share['id'])
        self.assertEqual(access_rule['state'], 'active')
        self.assertEqual(access_rule['access_type'], 'ip')
        self.assertEqual(access_rule['access_to'], '0.0.0.0/0')
        self.assertEqual(access_rule['properties'], '')
        self.assertEqual(access_rule['access_level'], 'rw')
        access_rules = self.listing_result('share', f'access list {share['id']}')
        self.assertIn(access_rule['id'], [item['ID'] for item in access_rules])
        access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='12.34.56.78', access_level='ro', properties='foo=bar')
        self.assertEqual(access_rule['access_type'], 'ip')
        self.assertEqual(access_rule['access_to'], '12.34.56.78')
        self.assertEqual(access_rule['properties'], 'foo : bar')
        self.assertEqual(access_rule['access_level'], 'ro')

    @ddt.data({'lock_visibility': True, 'lock_deletion': True, 'lock_reason': None}, {'lock_visibility': False, 'lock_deletion': True, 'lock_reason': None}, {'lock_visibility': True, 'lock_deletion': False, 'lock_reason': 'testing'}, {'lock_visibility': True, 'lock_deletion': False, 'lock_reason': 'testing'})
    @ddt.unpack
    def test_share_access_allow_restrict(self, lock_visibility, lock_deletion, lock_reason):
        share = self.create_share()
        access_rule = self.create_share_access_rule(share=share['id'], access_type='ip', access_to='0.0.0.0/0', wait=True, lock_visibility=lock_visibility, lock_deletion=lock_deletion, lock_reason=lock_reason)
        if lock_deletion:
            self.assertRaises(tempest_exc.CommandFailed, self.openstack, 'share', params=f'access delete {share['id']} {access_rule['id']}')
        self.openstack('share', params=f'access delete {share['id']} {access_rule['id']} --unrestrict --wait')