from openstack.network.v2 import vpn_service
from openstack.tests.unit import base
class TestVpnService(base.TestCase):

    def test_basic(self):
        sot = vpn_service.VpnService()
        self.assertEqual('vpnservice', sot.resource_key)
        self.assertEqual('vpnservices', sot.resources_key)
        self.assertEqual('/vpn/vpnservices', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = vpn_service.VpnService(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['external_v4_ip'], sot.external_v4_ip)
        self.assertEqual(EXAMPLE['external_v6_ip'], sot.external_v6_ip)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['router_id'], sot.router_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['subnet_id'], sot.subnet_id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'external_v4_ip': 'external_v4_ip', 'external_v6_ip': 'external_v6_ip', 'name': 'name', 'router_id': 'router_id', 'project_id': 'project_id', 'tenant_id': 'tenant_id', 'subnet_id': 'subnet_id', 'is_admin_state_up': 'admin_state_up'}, sot._query_mapping._mapping)