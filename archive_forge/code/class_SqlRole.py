import uuid
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.assignment import test_core
from keystone.tests.unit.backend import core_sql
class SqlRole(core_sql.BaseBackendSqlTests, test_core.RoleTests):

    def test_create_null_role_name(self):
        role = unit.new_role_ref(name=None)
        self.assertRaises(exception.UnexpectedError, PROVIDERS.role_api.create_role, role['id'], role)
        self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, role['id'])

    def test_create_duplicate_role_domain_specific_name_fails(self):
        domain = unit.new_domain_ref()
        role1 = unit.new_role_ref(domain_id=domain['id'])
        PROVIDERS.role_api.create_role(role1['id'], role1)
        role2 = unit.new_role_ref(name=role1['name'], domain_id=domain['id'])
        self.assertRaises(exception.Conflict, PROVIDERS.role_api.create_role, role2['id'], role2)

    def test_update_domain_id_of_role_fails(self):
        role1 = unit.new_role_ref()
        role1 = PROVIDERS.role_api.create_role(role1['id'], role1)
        domainA = unit.new_domain_ref()
        role1['domain_id'] = domainA['id']
        self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role1['id'], role1)
        role2 = unit.new_role_ref(domain_id=domainA['id'])
        PROVIDERS.role_api.create_role(role2['id'], role2)
        domainB = unit.new_domain_ref()
        role2['domain_id'] = domainB['id']
        self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role2['id'], role2)
        role2['domain_id'] = None
        self.assertRaises(exception.ValidationError, PROVIDERS.role_api.update_role, role2['id'], role2)

    def test_domain_specific_separation(self):
        domain1 = unit.new_domain_ref()
        role1 = unit.new_role_ref(domain_id=domain1['id'])
        role_ref1 = PROVIDERS.role_api.create_role(role1['id'], role1.copy())
        self.assertDictEqual(role1, role_ref1)
        domain2 = unit.new_domain_ref()
        role2 = unit.new_role_ref(name=role1['name'], domain_id=domain2['id'])
        role_ref2 = PROVIDERS.role_api.create_role(role2['id'], role2)
        self.assertDictEqual(role2, role_ref2)
        role3 = unit.new_role_ref(name=role1['name'])
        role_ref3 = PROVIDERS.role_api.create_role(role3['id'], role3)
        self.assertDictEqual(role3, role_ref3)
        role1['name'] = uuid.uuid4().hex
        PROVIDERS.role_api.update_role(role1['id'], role1)
        role_ref1 = PROVIDERS.role_api.get_role(role1['id'])
        self.assertDictEqual(role1, role_ref1)
        role_ref2 = PROVIDERS.role_api.get_role(role2['id'])
        self.assertDictEqual(role2, role_ref2)
        role_ref3 = PROVIDERS.role_api.get_role(role3['id'])
        self.assertDictEqual(role3, role_ref3)
        PROVIDERS.role_api.delete_role(role1['id'])
        self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, role1['id'])
        PROVIDERS.role_api.get_role(role2['id'])
        PROVIDERS.role_api.get_role(role3['id'])