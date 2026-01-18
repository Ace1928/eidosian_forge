from openstack.shared_file_system.v2 import share_network
from openstack.tests.unit import base
class TestShareNetwork(base.TestCase):

    def test_basic(self):
        networks = share_network.ShareNetwork()
        self.assertEqual('share_networks', networks.resources_key)
        self.assertEqual('/share-networks', networks.base_path)
        self.assertTrue(networks.allow_list)
        self.assertTrue(networks.allow_create)
        self.assertTrue(networks.allow_fetch)
        self.assertTrue(networks.allow_commit)
        self.assertTrue(networks.allow_delete)
        self.assertFalse(networks.allow_head)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'project_id': 'project_id', 'created_since': 'created_since', 'created_before': 'created_before', 'offset': 'offset', 'security_service_id': 'security_service_id', 'project_id': 'project_id', 'all_projects': 'all_tenants', 'name': 'name', 'description': 'description'}, networks._query_mapping._mapping)

    def test_share_network(self):
        networks = share_network.ShareNetwork(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], networks.id)
        self.assertEqual(EXAMPLE['name'], networks.name)
        self.assertEqual(EXAMPLE['project_id'], networks.project_id)
        self.assertEqual(EXAMPLE['description'], networks.description)
        self.assertEqual(EXAMPLE['created_at'], networks.created_at)
        self.assertEqual(EXAMPLE['updated_at'], networks.updated_at)