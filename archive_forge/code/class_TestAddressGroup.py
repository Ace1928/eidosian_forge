from openstack.network.v2 import address_group
from openstack.tests.unit import base
class TestAddressGroup(base.TestCase):

    def test_basic(self):
        sot = address_group.AddressGroup()
        self.assertEqual('address_group', sot.resource_key)
        self.assertEqual('address_groups', sot.resources_key)
        self.assertEqual('/address-groups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'name': 'name', 'description': 'description', 'project_id': 'project_id', 'sort_key': 'sort_key', 'sort_dir': 'sort_dir', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = address_group.AddressGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertCountEqual(EXAMPLE['addresses'], sot.addresses)