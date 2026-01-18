from openstack.network.v2 import security_group
from openstack.tests.unit import base
class TestSecurityGroup(base.TestCase):

    def test_basic(self):
        sot = security_group.SecurityGroup()
        self.assertEqual('security_group', sot.resource_key)
        self.assertEqual('security_groups', sot.resources_key)
        self.assertEqual('/security-groups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'any_tags': 'tags-any', 'description': 'description', 'fields': 'fields', 'id': 'id', 'limit': 'limit', 'marker': 'marker', 'name': 'name', 'not_any_tags': 'not-tags-any', 'not_tags': 'not-tags', 'tenant_id': 'tenant_id', 'revision_number': 'revision_number', 'sort_dir': 'sort_dir', 'sort_key': 'sort_key', 'tags': 'tags', 'project_id': 'project_id', 'stateful': 'stateful'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = security_group.SecurityGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['revision_number'], sot.revision_number)
        self.assertEqual(EXAMPLE['security_group_rules'], sot.security_group_rules)
        self.assertEqual(dict, type(sot.security_group_rules[0]))
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)
        self.assertEqual(EXAMPLE['tags'], sot.tags)