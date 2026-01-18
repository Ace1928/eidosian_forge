import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
class UnsetShareAccessRulesTestCase(base.OSCClientTestBase):

    def test_unset_share_access(self):
        share = self.create_share()
        access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='192.168.0.101', wait=True, properties='foo=bar')
        self.openstack('share', params=f'access unset --property foo {access_rule['id']}')
        access_rule_unset = self.dict_result('share', f'access show {access_rule['id']}')
        self.assertEqual(access_rule_unset['properties'], '')