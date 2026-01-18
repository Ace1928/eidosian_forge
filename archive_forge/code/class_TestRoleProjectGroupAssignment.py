from openstack.identity.v3 import role_project_group_assignment
from openstack.tests.unit import base
class TestRoleProjectGroupAssignment(base.TestCase):

    def test_basic(self):
        sot = role_project_group_assignment.RoleProjectGroupAssignment()
        self.assertEqual('role', sot.resource_key)
        self.assertEqual('roles', sot.resources_key)
        self.assertEqual('/projects/%(project_id)s/groups/%(group_id)s/roles', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = role_project_group_assignment.RoleProjectGroupAssignment(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['group_id'], sot.group_id)