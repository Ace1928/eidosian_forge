from openstack.identity.v3 import role_domain_user_assignment
from openstack.tests.unit import base
class TestRoleDomainUserAssignment(base.TestCase):

    def test_basic(self):
        sot = role_domain_user_assignment.RoleDomainUserAssignment()
        self.assertEqual('role', sot.resource_key)
        self.assertEqual('roles', sot.resources_key)
        self.assertEqual('/domains/%(domain_id)s/users/%(user_id)s/roles', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = role_domain_user_assignment.RoleDomainUserAssignment(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['domain_id'], sot.domain_id)
        self.assertEqual(EXAMPLE['user_id'], sot.user_id)