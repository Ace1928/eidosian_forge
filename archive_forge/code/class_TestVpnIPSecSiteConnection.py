from openstack.network.v2 import vpn_ipsec_site_connection
from openstack.tests.unit import base
class TestVpnIPSecSiteConnection(base.TestCase):

    def test_basic(self):
        sot = vpn_ipsec_site_connection.VpnIPSecSiteConnection()
        self.assertEqual('ipsec_site_connection', sot.resource_key)
        self.assertEqual('ipsec_site_connections', sot.resources_key)
        self.assertEqual('/vpn/ipsec-site-connections', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = vpn_ipsec_site_connection.VpnIPSecSiteConnection(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['auth_mode'], sot.auth_mode)
        self.assertEqual(EXAMPLE['ikepolicy_id'], sot.ikepolicy_id)
        self.assertEqual(EXAMPLE['vpnservice_id'], sot.vpnservice_id)
        self.assertEqual(EXAMPLE['local_ep_group_id'], sot.local_ep_group_id)
        self.assertEqual(EXAMPLE['peer_address'], sot.peer_address)
        self.assertEqual(EXAMPLE['route_mode'], sot.route_mode)
        self.assertEqual(EXAMPLE['ipsecpolicy_id'], sot.ipsecpolicy_id)
        self.assertEqual(EXAMPLE['peer_id'], sot.peer_id)
        self.assertEqual(EXAMPLE['psk'], sot.psk)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['initiator'], sot.initiator)
        self.assertEqual(EXAMPLE['peer_cidrs'], sot.peer_cidrs)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['tenant_id'], sot.project_id)
        self.assertEqual(EXAMPLE['interval'], sot.interval)
        self.assertEqual(EXAMPLE['mtu'], sot.mtu)
        self.assertEqual(EXAMPLE['peer_ep_group_id'], sot.peer_ep_group_id)
        self.assertEqual(EXAMPLE['dpd'], sot.dpd)
        self.assertEqual(EXAMPLE['timeout'], sot.timeout)
        self.assertEqual(EXAMPLE['action'], sot.action)
        self.assertEqual(EXAMPLE['local_id'], sot.local_id)