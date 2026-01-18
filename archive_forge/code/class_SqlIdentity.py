import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
class SqlIdentity(SqlTests, identity_tests.IdentityTests, assignment_tests.AssignmentTests, assignment_tests.SystemAssignmentTests, resource_tests.ResourceTests):

    def test_password_hashed(self):
        with sql.session_for_read() as session:
            user_ref = PROVIDERS.identity_api._get_user(session, self.user_foo['id'])
            self.assertNotEqual(self.user_foo['password'], user_ref['password'])

    def test_create_user_with_null_password(self):
        user_dict = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_dict['password'] = None
        new_user_dict = PROVIDERS.identity_api.create_user(user_dict)
        with sql.session_for_read() as session:
            new_user_ref = PROVIDERS.identity_api._get_user(session, new_user_dict['id'])
            self.assertIsNone(new_user_ref.password)

    def test_update_user_with_null_password(self):
        user_dict = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        self.assertTrue(user_dict['password'])
        new_user_dict = PROVIDERS.identity_api.create_user(user_dict)
        new_user_dict['password'] = None
        new_user_dict = PROVIDERS.identity_api.update_user(new_user_dict['id'], new_user_dict)
        with sql.session_for_read() as session:
            new_user_ref = PROVIDERS.identity_api._get_user(session, new_user_dict['id'])
            self.assertIsNone(new_user_ref.password)

    def test_delete_user_with_project_association(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_bar['id'], role_member['id'])
        PROVIDERS.identity_api.delete_user(user['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, user['id'])

    def test_create_user_case_sensitivity(self):
        ref = unit.new_user_ref(name=uuid.uuid4().hex.lower(), domain_id=CONF.identity.default_domain_id)
        ref = PROVIDERS.identity_api.create_user(ref)
        ref['name'] = ref['name'].upper()
        PROVIDERS.identity_api.create_user(ref)

    def test_create_project_case_sensitivity(self):
        ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(ref['id'], ref)
        ref['id'] = uuid.uuid4().hex
        ref['name'] = ref['name'].upper()
        PROVIDERS.resource_api.create_project(ref['id'], ref)

    def test_delete_project_with_user_association(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_bar['id'], role_member['id'])
        PROVIDERS.resource_api.delete_project(self.project_bar['id'])
        projects = PROVIDERS.assignment_api.list_projects_for_user(user['id'])
        self.assertEqual([], projects)

    def test_update_project_returns_extra(self):
        """Test for backward compatibility with an essex/folsom bug.

        Non-indexed attributes were returned in an 'extra' attribute, instead
        of on the entity itself; for consistency and backwards compatibility,
        those attributes should be included twice.

        This behavior is specific to the SQL driver.

        """
        arbitrary_key = uuid.uuid4().hex
        arbitrary_value = uuid.uuid4().hex
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project[arbitrary_key] = arbitrary_value
        ref = PROVIDERS.resource_api.create_project(project['id'], project)
        self.assertEqual(arbitrary_value, ref[arbitrary_key])
        self.assertNotIn('extra', ref)
        ref['name'] = uuid.uuid4().hex
        ref = PROVIDERS.resource_api.update_project(ref['id'], ref)
        self.assertEqual(arbitrary_value, ref[arbitrary_key])
        self.assertEqual(arbitrary_value, ref['extra'][arbitrary_key])

    def test_update_user_returns_extra(self):
        """Test for backwards-compatibility with an essex/folsom bug.

        Non-indexed attributes were returned in an 'extra' attribute, instead
        of on the entity itself; for consistency and backwards compatibility,
        those attributes should be included twice.

        This behavior is specific to the SQL driver.

        """
        arbitrary_key = uuid.uuid4().hex
        arbitrary_value = uuid.uuid4().hex
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user[arbitrary_key] = arbitrary_value
        del user['id']
        ref = PROVIDERS.identity_api.create_user(user)
        self.assertEqual(arbitrary_value, ref[arbitrary_key])
        self.assertNotIn('password', ref)
        self.assertNotIn('extra', ref)
        user['name'] = uuid.uuid4().hex
        user['password'] = uuid.uuid4().hex
        ref = PROVIDERS.identity_api.update_user(ref['id'], user)
        self.assertNotIn('password', ref)
        self.assertNotIn('password', ref['extra'])
        self.assertEqual(arbitrary_value, ref[arbitrary_key])
        self.assertEqual(arbitrary_value, ref['extra'][arbitrary_key])

    def test_sql_user_to_dict_null_default_project_id(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        with sql.session_for_read() as session:
            query = session.query(identity_sql.User)
            query = query.filter_by(id=user['id'])
            raw_user_ref = query.one()
            self.assertIsNone(raw_user_ref.default_project_id)
            user_ref = raw_user_ref.to_dict()
            self.assertNotIn('default_project_id', user_ref)
            session.close()

    def test_list_domains_for_user(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.new_user_ref(domain_id=domain['id'])
        test_domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(test_domain1['id'], test_domain1)
        test_domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(test_domain2['id'], test_domain2)
        user = PROVIDERS.identity_api.create_user(user)
        user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
        self.assertEqual(0, len(user_domains))
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=test_domain1['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=test_domain2['id'], role_id=self.role_member['id'])
        user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
        self.assertThat(user_domains, matchers.HasLength(2))

    def test_list_domains_for_user_with_grants(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.new_user_ref(domain_id=domain['id'])
        user = PROVIDERS.identity_api.create_user(user)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domain['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        test_domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(test_domain1['id'], test_domain1)
        test_domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(test_domain2['id'], test_domain2)
        test_domain3 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(test_domain3['id'], test_domain3)
        PROVIDERS.identity_api.add_user_to_group(user['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user['id'], group2['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=test_domain1['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=test_domain2['id'], role_id=self.role_admin['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=test_domain3['id'], role_id=self.role_admin['id'])
        user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
        self.assertThat(user_domains, matchers.HasLength(3))

    def test_list_domains_for_user_with_inherited_grants(self):
        """Test that inherited roles on the domain are excluded.

        Test Plan:

        - Create two domains, one user, group and role
        - Domain1 is given an inherited user role, Domain2 an inherited
          group role (for a group of which the user is a member)
        - When listing domains for user, neither domain should be returned

        """
        domain1 = unit.new_domain_ref()
        domain1 = PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        domain2 = PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        user = unit.new_user_ref(domain_id=domain1['id'])
        user = PROVIDERS.identity_api.create_user(user)
        group = unit.new_group_ref(domain_id=domain1['id'])
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=domain1['id'], role_id=role['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group['id'], domain_id=domain2['id'], role_id=role['id'], inherited_to_projects=True)
        user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
        self.assertThat(user_domains, matchers.HasLength(0))

    def test_list_groups_for_user(self):
        domain = self._get_domain_fixture()
        test_groups = []
        test_users = []
        GROUP_COUNT = 3
        USER_COUNT = 2
        for x in range(0, USER_COUNT):
            new_user = unit.new_user_ref(domain_id=domain['id'])
            new_user = PROVIDERS.identity_api.create_user(new_user)
            test_users.append(new_user)
        positive_user = test_users[0]
        negative_user = test_users[1]
        for x in range(0, USER_COUNT):
            group_refs = PROVIDERS.identity_api.list_groups_for_user(test_users[x]['id'])
            self.assertEqual(0, len(group_refs))
        for x in range(0, GROUP_COUNT):
            before_count = x
            after_count = x + 1
            new_group = unit.new_group_ref(domain_id=domain['id'])
            new_group = PROVIDERS.identity_api.create_group(new_group)
            test_groups.append(new_group)
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(before_count, len(group_refs))
            PROVIDERS.identity_api.add_user_to_group(positive_user['id'], new_group['id'])
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(after_count, len(group_refs))
            group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
            self.assertEqual(0, len(group_refs))
        for x in range(0, 3):
            before_count = GROUP_COUNT - x
            after_count = GROUP_COUNT - x - 1
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(before_count, len(group_refs))
            PROVIDERS.identity_api.remove_user_from_group(positive_user['id'], test_groups[x]['id'])
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(after_count, len(group_refs))
            group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
            self.assertEqual(0, len(group_refs))

    def test_add_user_to_group_expiring_mapped(self):
        self._build_fed_resource()
        domain = self._get_domain_fixture()
        self.config_fixture.config(group='federation', default_authorization_ttl=5)
        time = datetime.datetime.utcnow()
        tick = datetime.timedelta(minutes=5)
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        fed_dict = unit.new_federated_user_ref()
        fed_dict['id'] = fed_dict['unique_id']
        fed_dict['name'] = fed_dict['display_name']
        fed_dict['domain'] = {'id': uuid.uuid4().hex}
        fed_dict['idp_id'] = 'myidp'
        fed_dict['protocol_id'] = 'mapped'
        with freezegun.freeze_time(time - tick) as frozen_time:
            user = PROVIDERS.identity_api.shadow_federated_user(fed_dict['idp_id'], fed_dict['protocol_id'], fed_dict, group_ids=[new_group['id']])
            PROVIDERS.identity_api.check_user_in_group(user['id'], new_group['id'])
            frozen_time.tick(tick)
            self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, user['id'], new_group['id'])
            PROVIDERS.identity_api.shadow_federated_user(fed_dict['idp_id'], fed_dict['protocol_id'], fed_dict, group_ids=[new_group['id']])
            PROVIDERS.identity_api.check_user_in_group(user['id'], new_group['id'])

    def test_add_user_to_group_expiring(self):
        self._build_fed_resource()
        domain = self._get_domain_fixture()
        time = datetime.datetime.utcnow()
        tick = datetime.timedelta(minutes=5)
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        fed_dict = unit.new_federated_user_ref()
        fed_dict['idp_id'] = 'myidp'
        fed_dict['protocol_id'] = 'mapped'
        new_user = PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict)
        with freezegun.freeze_time(time - tick) as frozen_time:
            PROVIDERS.shadow_users_api.add_user_to_group_expires(new_user['id'], new_group['id'])
            self.config_fixture.config(group='federation', default_authorization_ttl=0)
            self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, new_user['id'], new_group['id'])
            self.config_fixture.config(group='federation', default_authorization_ttl=5)
            PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])
            frozen_time.tick(tick)
            self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, new_user['id'], new_group['id'])
            PROVIDERS.shadow_users_api.add_user_to_group_expires(new_user['id'], new_group['id'])
            PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])

    def test_add_user_to_group_expiring_list(self):
        self._build_fed_resource()
        domain = self._get_domain_fixture()
        self.config_fixture.config(group='federation', default_authorization_ttl=5)
        time = datetime.datetime.utcnow()
        tick = datetime.timedelta(minutes=5)
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        exp_new_group = unit.new_group_ref(domain_id=domain['id'])
        exp_new_group = PROVIDERS.identity_api.create_group(exp_new_group)
        fed_dict = unit.new_federated_user_ref()
        fed_dict['idp_id'] = 'myidp'
        fed_dict['protocol_id'] = 'mapped'
        new_user = PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])
        with freezegun.freeze_time(time - tick) as frozen_time:
            PROVIDERS.shadow_users_api.add_user_to_group_expires(new_user['id'], exp_new_group['id'])
            PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])
            groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
            self.assertEqual(len(groups), 2)
            for group in groups:
                if group.get('membership_expires_at'):
                    self.assertEqual(group['membership_expires_at'], time)
            frozen_time.tick(tick)
            groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
            self.assertEqual(len(groups), 1)

    def test_storing_null_domain_id_in_project_ref(self):
        """Test the special storage of domain_id=None in sql resource driver.

        The resource driver uses a special value in place of None for domain_id
        in the project record. This shouldn't escape the driver. Hence we test
        the interface to ensure that you can store a domain_id of None, and
        that any special value used inside the driver does not escape through
        the interface.

        """
        spoiler_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(spoiler_project['id'], spoiler_project)
        project = unit.new_project_ref(domain_id=None, is_domain=True)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertDictEqual(project, ref)
        ref = PROVIDERS.resource_api.get_project_by_name(project['name'], None)
        self.assertDictEqual(project, ref)
        project2 = unit.new_project_ref(domain_id=None, is_domain=True)
        project2 = PROVIDERS.resource_api.create_project(project2['id'], project2)
        hints = driver_hints.Hints()
        hints.add_filter('domain_id', None)
        refs = PROVIDERS.resource_api.list_projects(hints)
        self.assertThat(refs, matchers.HasLength(2 + self.domain_count))
        self.assertIn(project, refs)
        self.assertIn(project2, refs)
        project['name'] = uuid.uuid4().hex
        PROVIDERS.resource_api.update_project(project['id'], project)
        ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertDictEqual(project, ref)
        project['enabled'] = False
        PROVIDERS.resource_api.update_project(project['id'], project)
        PROVIDERS.resource_api.delete_project(project['id'])
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project['id'])

    def test_hidden_project_domain_root_is_really_hidden(self):
        """Ensure we cannot access the hidden root of all project domains.

        Calling any of the driver methods should result in the same as
        would be returned if we passed a project that does not exist. We don't
        test create_project, since we do not allow a caller of our API to
        specify their own ID for a new entity.

        """

        def _exercise_project_api(ref_id):
            driver = PROVIDERS.resource_api.driver
            self.assertRaises(exception.ProjectNotFound, driver.get_project, ref_id)
            self.assertRaises(exception.ProjectNotFound, driver.get_project_by_name, resource.NULL_DOMAIN_ID, ref_id)
            project_ids = [x['id'] for x in driver.list_projects(driver_hints.Hints())]
            self.assertNotIn(ref_id, project_ids)
            projects = driver.list_projects_from_ids([ref_id])
            self.assertThat(projects, matchers.HasLength(0))
            project_ids = [x for x in driver.list_project_ids_from_domain_ids([ref_id])]
            self.assertNotIn(ref_id, project_ids)
            self.assertRaises(exception.DomainNotFound, driver.list_projects_in_domain, ref_id)
            project_ids = [x['id'] for x in driver.list_projects_acting_as_domain(driver_hints.Hints())]
            self.assertNotIn(ref_id, project_ids)
            projects = driver.list_projects_in_subtree(ref_id)
            self.assertThat(projects, matchers.HasLength(0))
            self.assertRaises(exception.ProjectNotFound, driver.list_project_parents, ref_id)
            self.assertTrue(driver.is_leaf_project(ref_id))
            self.assertRaises(exception.ProjectNotFound, driver.update_project, ref_id, {})
            self.assertRaises(exception.ProjectNotFound, driver.delete_project, ref_id)
            if ref_id != resource.NULL_DOMAIN_ID:
                driver.delete_projects_from_ids([ref_id])
        _exercise_project_api(uuid.uuid4().hex)
        _exercise_project_api(resource.NULL_DOMAIN_ID)

    def test_list_users_call_count(self):
        """There should not be O(N) queries."""
        for i in range(10):
            user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
            PROVIDERS.identity_api.create_user(user)

        class CallCounter(object):

            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls = 0

            def query_counter(self, query):
                self.calls += 1
        counter = CallCounter()
        sqlalchemy.event.listen(sqlalchemy.orm.query.Query, 'before_compile', counter.query_counter)
        first_call_users = PROVIDERS.identity_api.list_users()
        first_call_counter = counter.calls
        for i in range(10):
            user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
            PROVIDERS.identity_api.create_user(user)
        counter.reset()
        second_call_users = PROVIDERS.identity_api.list_users()
        self.assertNotEqual(len(first_call_users), len(second_call_users))
        self.assertEqual(first_call_counter, counter.calls)
        self.assertEqual(3, counter.calls)

    def test_check_project_depth(self):
        project_1 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_1['id'], project_1)
        project_2 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=project_1['id'])
        PROVIDERS.resource_api.create_project(project_2['id'], project_2)
        resp = PROVIDERS.resource_api.check_project_depth(max_depth=None)
        self.assertIsNone(resp)
        resp = PROVIDERS.resource_api.check_project_depth(max_depth=3)
        self.assertIsNone(resp)
        resp = PROVIDERS.resource_api.check_project_depth(max_depth=4)
        self.assertIsNone(resp)
        self.assertRaises(exception.LimitTreeExceedError, PROVIDERS.resource_api.check_project_depth, 2)

    def test_update_user_with_stale_data_forces_retry(self):
        log_fixture = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        user_dict = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        new_user_dict = PROVIDERS.identity_api.create_user(user_dict)
        side_effects = [sqlalchemy.orm.exc.StaleDataError, sql.session_for_write()]
        with mock.patch('keystone.common.sql.session_for_write') as m:
            m.side_effect = side_effects
            new_user_dict['email'] = uuid.uuid4().hex
            PROVIDERS.identity_api.update_user(new_user_dict['id'], new_user_dict)
        expected_log_message = 'Performing DB retry for function keystone.identity.backends.sql.Identity.update_user'
        self.assertIn(expected_log_message, log_fixture.output)