from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _test_list_role_assignment_containing_names(self, domain_role=False):
    new_domain = self._get_domain_fixture()
    if domain_role:
        new_role = unit.new_role_ref(domain_id=new_domain['id'])
    else:
        new_role = unit.new_role_ref()
    new_user = unit.new_user_ref(domain_id=new_domain['id'])
    new_project = unit.new_project_ref(domain_id=new_domain['id'])
    new_group = unit.new_group_ref(domain_id=new_domain['id'])
    new_role = PROVIDERS.role_api.create_role(new_role['id'], new_role)
    new_user = PROVIDERS.identity_api.create_user(new_user)
    new_group = PROVIDERS.identity_api.create_group(new_group)
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=new_role['id'])
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=new_role['id'])
    PROVIDERS.assignment_api.create_grant(domain_id=new_domain['id'], user_id=new_user['id'], role_id=new_role['id'])
    _asgmt_prj = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'], include_names=True)
    _asgmt_grp = PROVIDERS.assignment_api.list_role_assignments(group_id=new_group['id'], project_id=new_project['id'], include_names=True)
    _asgmt_dmn = PROVIDERS.assignment_api.list_role_assignments(domain_id=new_domain['id'], user_id=new_user['id'], include_names=True)
    self.assertThat(_asgmt_prj, matchers.HasLength(1))
    self.assertThat(_asgmt_grp, matchers.HasLength(1))
    self.assertThat(_asgmt_dmn, matchers.HasLength(1))
    first_asgmt_prj = _asgmt_prj[0]
    first_asgmt_grp = _asgmt_grp[0]
    first_asgmt_dmn = _asgmt_dmn[0]
    self.assertEqual(new_project['name'], first_asgmt_prj['project_name'])
    self.assertEqual(new_project['domain_id'], first_asgmt_prj['project_domain_id'])
    self.assertEqual(new_user['name'], first_asgmt_prj['user_name'])
    self.assertEqual(new_user['domain_id'], first_asgmt_prj['user_domain_id'])
    self.assertEqual(new_role['name'], first_asgmt_prj['role_name'])
    if domain_role:
        self.assertEqual(new_role['domain_id'], first_asgmt_prj['role_domain_id'])
    self.assertEqual(new_group['name'], first_asgmt_grp['group_name'])
    self.assertEqual(new_group['domain_id'], first_asgmt_grp['group_domain_id'])
    self.assertEqual(new_project['name'], first_asgmt_grp['project_name'])
    self.assertEqual(new_project['domain_id'], first_asgmt_grp['project_domain_id'])
    self.assertEqual(new_role['name'], first_asgmt_grp['role_name'])
    if domain_role:
        self.assertEqual(new_role['domain_id'], first_asgmt_grp['role_domain_id'])
    self.assertEqual(new_domain['name'], first_asgmt_dmn['domain_name'])
    self.assertEqual(new_user['name'], first_asgmt_dmn['user_name'])
    self.assertEqual(new_user['domain_id'], first_asgmt_dmn['user_domain_id'])
    self.assertEqual(new_role['name'], first_asgmt_dmn['role_name'])
    if domain_role:
        self.assertEqual(new_role['domain_id'], first_asgmt_dmn['role_domain_id'])