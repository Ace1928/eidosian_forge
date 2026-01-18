import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
class BaseLDAPIdentity(LDAPTestSetup, IdentityTests, AssignmentTests, ResourceTests):

    def _get_domain_fixture(self):
        """Return the static domain, since domains in LDAP are read-only."""
        return PROVIDERS.resource_api.get_domain(CONF.identity.default_domain_id)

    def get_config(self, domain_id):
        return CONF

    def config_overrides(self):
        super(BaseLDAPIdentity, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')

    def config_files(self):
        config_files = super(BaseLDAPIdentity, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap.conf'))
        return config_files

    def new_user_ref(self, domain_id, project_id=None, **kwargs):
        ref = unit.new_user_ref(domain_id=domain_id, project_id=project_id, **kwargs)
        if 'id' not in kwargs:
            del ref['id']
        return ref

    def get_user_enabled_vals(self, user):
        user_dn = PROVIDERS.identity_api.driver.user._id_to_dn_string(user['id'])
        enabled_attr_name = CONF.ldap.user_enabled_attribute
        ldap_ = PROVIDERS.identity_api.driver.user.get_connection()
        res = ldap_.search_s(user_dn, ldap.SCOPE_BASE, u'(sn=%s)' % user['name'])
        if enabled_attr_name in res[0][1]:
            return res[0][1][enabled_attr_name]
        else:
            return None

    def test_build_tree(self):
        """Regression test for building the tree names."""
        user_api = identity.backends.ldap.UserApi(CONF)
        self.assertTrue(user_api)
        self.assertEqual('ou=Users,%s' % CONF.ldap.suffix, user_api.tree_dn)

    def test_configurable_allowed_user_actions(self):
        user = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.get_user(user['id'])
        user['password'] = u'fäképass2'
        PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertRaises(exception.Forbidden, PROVIDERS.identity_api.delete_user, user['id'])

    def test_user_filter(self):
        user_ref = PROVIDERS.identity_api.get_user(self.user_foo['id'])
        self.user_foo.pop('password')
        self.assertDictEqual(self.user_foo, user_ref)
        driver = PROVIDERS.identity_api._select_identity_driver(user_ref['domain_id'])
        driver.user.ldap_filter = '(CN=DOES_NOT_MATCH)'
        PROVIDERS.identity_api.get_user.invalidate(PROVIDERS.identity_api, self.user_foo['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, self.user_foo['id'])

    def test_list_users_by_name_and_with_filter(self):
        hints = driver_hints.Hints()
        hints.add_filter('name', self.user_foo['name'])
        domain_id = self.user_foo['domain_id']
        driver = PROVIDERS.identity_api._select_identity_driver(domain_id)
        driver.user.ldap_filter = '(|(cn=%s)(cn=%s))' % (self.user_sna['id'], self.user_two['id'])
        users = PROVIDERS.identity_api.list_users(domain_scope=self._set_domain_scope(domain_id), hints=hints)
        self.assertEqual(0, len(users))

    def test_list_groups_by_name_and_with_filter(self):
        domain = self._get_domain_fixture()
        group_names = []
        numgroups = 3
        for _ in range(numgroups):
            group = unit.new_group_ref(domain_id=domain['id'])
            group = PROVIDERS.identity_api.create_group(group)
            group_names.append(group['name'])
        groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']))
        self.assertEqual(numgroups, len(groups))
        driver = PROVIDERS.identity_api._select_identity_driver(domain['id'])
        driver.group.ldap_filter = '(|(ou=%s)(ou=%s))' % tuple(group_names[:2])
        groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']))
        self.assertEqual(2, len(groups))
        hints = driver_hints.Hints()
        hints.add_filter('name', group_names[2])
        groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']), hints=hints)
        self.assertEqual(0, len(groups))

    def test_remove_role_grant_from_user_and_project(self):
        PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_group_and_project(self):
        new_domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = self.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertEqual([], roles_ref)
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertNotEmpty(roles_ref)
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_group_and_domain(self):
        new_domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = self.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.NotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_list_projects_for_user(self):
        domain = self._get_domain_fixture()
        user1 = self.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertThat(user_projects, matchers.HasLength(0))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_baz['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertThat(user_projects, matchers.HasLength(2))
        user2 = self.new_user_ref(domain_id=domain['id'])
        user2 = PROVIDERS.identity_api.create_user(user2)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=self.project_baz['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user2['id'])
        self.assertThat(user_projects, matchers.HasLength(2))
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=self.project_bar['id'], role_id=self.role_other['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user2['id'])
        self.assertThat(user_projects, matchers.HasLength(2))

    def test_list_projects_for_user_and_groups(self):
        domain = self._get_domain_fixture()
        user1 = self.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertThat(user_projects, matchers.HasLength(1))
        PROVIDERS.assignment_api.delete_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertThat(user_projects, matchers.HasLength(1))

    def test_list_projects_for_user_with_grants(self):
        domain = self._get_domain_fixture()
        new_user = self.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domain['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], group2['id'])
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=project1['id'], role_id=self.role_admin['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], project_id=project2['id'], role_id=self.role_admin['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(new_user['id'])
        self.assertEqual(3, len(user_projects))

    def test_list_role_assignments_unfiltered(self):
        new_domain = self._get_domain_fixture()
        new_user = self.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_project = unit.new_project_ref(domain_id=new_domain['id'])
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        existing_assignments = len(PROVIDERS.assignment_api.list_role_assignments())
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=default_fixtures.OTHER_ROLE_ID)
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=default_fixtures.ADMIN_ROLE_ID)
        after_assignments = len(PROVIDERS.assignment_api.list_role_assignments())
        self.assertEqual(existing_assignments + 2, after_assignments)

    def test_list_group_members_when_no_members(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.list_users_in_group(group['id'])

    def test_list_domains(self):
        domain1 = unit.new_domain_ref()
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        domains = PROVIDERS.resource_api.list_domains()
        self.assertEqual(7, len(domains))
        domain_ids = []
        for domain in domains:
            domain_ids.append(domain.get('id'))
        self.assertIn(CONF.identity.default_domain_id, domain_ids)
        self.assertIn(domain1['id'], domain_ids)
        self.assertIn(domain2['id'], domain_ids)

    def test_authenticate_requires_simple_bind(self):
        user = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_baz['id'], role_member['id'])
        driver = PROVIDERS.identity_api._select_identity_driver(user['domain_id'])
        driver.user.LDAP_USER = None
        driver.user.LDAP_PASSWORD = None
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=None)

    @mock.patch.object(versionutils, 'report_deprecated_feature')
    def test_user_crud(self, mock_deprecator):
        user_dict = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user_dict)
        args, _kwargs = mock_deprecator.call_args
        self.assertIn('create_user for the LDAP identity backend', args[1])
        del user_dict['password']
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        user_ref_dict = {x: user_ref[x] for x in user_ref}
        self.assertLessEqual(user_dict.items(), user_ref_dict.items())
        user_dict['password'] = uuid.uuid4().hex
        PROVIDERS.identity_api.update_user(user['id'], user_dict)
        args, _kwargs = mock_deprecator.call_args
        self.assertIn('update_user for the LDAP identity backend', args[1])
        del user_dict['password']
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        user_ref_dict = {x: user_ref[x] for x in user_ref}
        self.assertLessEqual(user_dict.items(), user_ref_dict.items())

    @mock.patch.object(versionutils, 'report_deprecated_feature')
    def test_group_crud(self, mock_deprecator):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        args, _kwargs = mock_deprecator.call_args
        self.assertIn('create_group for the LDAP identity backend', args[1])
        group_ref = PROVIDERS.identity_api.get_group(group['id'])
        self.assertDictEqual(group, group_ref)
        group['description'] = uuid.uuid4().hex
        PROVIDERS.identity_api.update_group(group['id'], group)
        args, _kwargs = mock_deprecator.call_args
        self.assertIn('update_group for the LDAP identity backend', args[1])
        group_ref = PROVIDERS.identity_api.get_group(group['id'])
        self.assertDictEqual(group, group_ref)

    @mock.patch.object(versionutils, 'report_deprecated_feature')
    def test_add_user_group_deprecated(self, mock_deprecator):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        args, _kwargs = mock_deprecator.call_args
        self.assertIn('add_user_to_group for the LDAP identity', args[1])

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_group_crud(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.get_group(group['id'])
        group['description'] = uuid.uuid4().hex
        group_ref = PROVIDERS.identity_api.update_group(group['id'], group)
        self.assertLessEqual(PROVIDERS.identity_api.get_group(group['id']).items(), group_ref.items())

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_get_user(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        user['description'] = uuid.uuid4().hex
        PROVIDERS.identity_api.get_user(ref['id'])
        user_updated = PROVIDERS.identity_api.update_user(ref['id'], user)
        self.assertLessEqual(PROVIDERS.identity_api.get_user(ref['id']).items(), user_updated.items())
        self.assertLessEqual(PROVIDERS.identity_api.get_user_by_name(ref['name'], ref['domain_id']).items(), user_updated.items())

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_get_user_by_name(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        user['description'] = uuid.uuid4().hex
        user_updated = PROVIDERS.identity_api.update_user(ref['id'], user)
        self.assertLessEqual(PROVIDERS.identity_api.get_user(ref['id']).items(), user_updated.items())
        self.assertLessEqual(PROVIDERS.identity_api.get_user_by_name(ref['name'], ref['domain_id']).items(), user_updated.items())

    def test_create_user_none_mapping(self):
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        driver.user.attribute_ignore = ['enabled', 'email', 'projects', 'projectId']
        user = self.new_user_ref(domain_id=CONF.identity.default_domain_id, project_id='maps_to_none')
        user = PROVIDERS.identity_api.create_user(user)

    def test_unignored_user_none_mapping(self):
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        driver.user.attribute_ignore = ['enabled', 'email', 'projects', 'projectId']
        user = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.get_user(user_ref['id'])

    def test_update_user_name(self):
        """A user's name cannot be changed through the LDAP driver."""
        self.assertRaises(exception.Conflict, super(BaseLDAPIdentity, self).test_update_user_name)

    def test_user_id_comma(self):
        """Even if the user has a , in their ID, groups can be listed."""
        user_id = u'Doe, John'
        user = self.new_user_ref(id=user_id, domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.driver.create_user(user_id, user)
        ref_list = PROVIDERS.identity_api.list_users()
        public_user_id = None
        for ref in ref_list:
            if ref['name'] == user['name']:
                public_user_id = ref['id']
                break
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_id = group['id']
        group = PROVIDERS.identity_api.driver.create_group(group_id, group)
        ref_list = PROVIDERS.identity_api.list_groups()
        public_group_id = None
        for ref in ref_list:
            if ref['name'] == group['name']:
                public_group_id = ref['id']
                break
        PROVIDERS.identity_api.add_user_to_group(public_user_id, public_group_id)
        ref_list = PROVIDERS.identity_api.list_groups_for_user(public_user_id)
        for ref in ref_list:
            del ref['membership_expires_at']
        group['id'] = public_group_id
        self.assertThat(ref_list, matchers.Equals([group]))

    def test_user_id_comma_grants(self):
        """List user and group grants, even with a comma in the user's ID."""
        user_id = u'Doe, John'
        user = self.new_user_ref(id=user_id, domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.driver.create_user(user_id, user)
        ref_list = PROVIDERS.identity_api.list_users()
        public_user_id = None
        for ref in ref_list:
            if ref['name'] == user['name']:
                public_user_id = ref['id']
                break
        role_id = default_fixtures.MEMBER_ROLE_ID
        project_id = self.project_baz['id']
        PROVIDERS.assignment_api.create_grant(role_id, user_id=public_user_id, project_id=project_id)
        role_ref = PROVIDERS.assignment_api.get_grant(role_id, user_id=public_user_id, project_id=project_id)
        self.assertEqual(role_id, role_ref['id'])

    def test_user_enabled_ignored_disable_error(self):
        self.config_fixture.config(group='ldap', user_attribute_ignore=['enabled'])
        self.load_backends()
        self.assertRaises(exception.ForbiddenAction, PROVIDERS.identity_api.update_user, self.user_foo['id'], {'enabled': False})
        user_info = PROVIDERS.identity_api.get_user(self.user_foo['id'])
        self.assertNotIn('enabled', user_info)

    def test_group_enabled_ignored_disable_error(self):
        self.config_fixture.config(group='ldap', group_attribute_ignore=['enabled'])
        self.load_backends()
        new_domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        self.assertRaises(exception.ForbiddenAction, PROVIDERS.identity_api.update_group, new_group['id'], {'enabled': False})
        group_info = PROVIDERS.identity_api.get_group(new_group['id'])
        self.assertNotIn('enabled', group_info)

    def test_list_role_assignment_by_domain(self):
        """Multiple domain assignments are not supported."""
        self.assertRaises((exception.Forbidden, exception.DomainNotFound, exception.ValidationError), super(BaseLDAPIdentity, self).test_list_role_assignment_by_domain)

    def test_list_role_assignment_by_user_with_domain_group_roles(self):
        """Multiple domain assignments are not supported."""
        self.assertRaises((exception.Forbidden, exception.DomainNotFound, exception.ValidationError), super(BaseLDAPIdentity, self).test_list_role_assignment_by_user_with_domain_group_roles)

    def test_list_role_assignment_using_sourced_groups_with_domains(self):
        """Multiple domain assignments are not supported."""
        self.assertRaises((exception.Forbidden, exception.ValidationError, exception.DomainNotFound), super(BaseLDAPIdentity, self).test_list_role_assignment_using_sourced_groups_with_domains)

    def test_create_project_with_domain_id_and_without_parent_id(self):
        """Multiple domains are not supported."""
        self.assertRaises(exception.ValidationError, super(BaseLDAPIdentity, self).test_create_project_with_domain_id_and_without_parent_id)

    def test_create_project_with_domain_id_mismatch_to_parent_domain(self):
        """Multiple domains are not supported."""
        self.assertRaises(exception.ValidationError, super(BaseLDAPIdentity, self).test_create_project_with_domain_id_mismatch_to_parent_domain)

    def test_remove_foreign_assignments_when_deleting_a_domain(self):
        """Multiple domains are not supported."""
        self.assertRaises((exception.ValidationError, exception.DomainNotFound), super(BaseLDAPIdentity, self).test_remove_foreign_assignments_when_deleting_a_domain)