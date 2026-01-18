from openstack.compute.v2 import server_group
from openstack.tests.unit import base
class TestServerGroup(base.TestCase):

    def test_basic(self):
        sot = server_group.ServerGroup()
        self.assertEqual('server_group', sot.resource_key)
        self.assertEqual('server_groups', sot.resources_key)
        self.assertEqual('/os-server-groups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'all_projects': 'all_projects', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = server_group.ServerGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['members'], sot.member_ids)
        self.assertEqual(EXAMPLE['metadata'], sot.metadata)
        self.assertEqual(EXAMPLE['policies'], sot.policies)
        self.assertEqual(EXAMPLE['rules'], sot.rules)