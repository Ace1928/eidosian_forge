from openstack.identity.v3 import role_system_group_assignment
from openstack.tests.unit import base
class TestRoleSystemGroupAssignment(base.TestCase):

    def test_basic(self):
        sot = role_system_group_assignment.RoleSystemGroupAssignment()
        self.assertEqual('role', sot.resource_key)
        self.assertEqual('roles', sot.resources_key)
        self.assertEqual('/system/groups/%(group_id)s/roles', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = role_system_group_assignment.RoleSystemGroupAssignment(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['group_id'], sot.group_id)