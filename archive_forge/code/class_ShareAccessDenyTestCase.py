import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
@ddt.ddt
class ShareAccessDenyTestCase(base.OSCClientTestBase):

    @ddt.data(True, False)
    def test_share_access_deny(self, lock_deletion):
        share = self.create_share()
        access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True, lock_deletion=lock_deletion)
        access_rules = self.listing_result('share', f'access list {share['id']}')
        num_access_rules = len(access_rules)
        delete_params = f'access delete {share['name']} {access_rule['id']} --wait'
        if lock_deletion:
            delete_params += ' --unrestrict'
        self.openstack('share', params=delete_params)
        access_rules = self.listing_result('share', f'access list {share['id']}')
        self.assertEqual(num_access_rules - 1, len(access_rules))