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
class DomainSpecificLDAPandSQLIdentity(BaseLDAPIdentity, unit.SQLDriverOverrides, unit.TestCase, BaseMultiLDAPandSQLIdentity):
    """Class to test when all domains use specific configs, including SQL.

    We define a set of domains and domain-specific backends:

    - A separate LDAP backend for the default domain
    - A separate SQL backend for domain1

    Although the default driver still exists, we don't use it.

    """
    DOMAIN_COUNT = 2
    DOMAIN_SPECIFIC_COUNT = 2

    def setUp(self):
        self.domain_count = self.DOMAIN_COUNT
        self.domain_specific_count = self.DOMAIN_SPECIFIC_COUNT
        super(DomainSpecificLDAPandSQLIdentity, self).setUp()

    def load_fixtures(self, fixtures):
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        self.setup_initial_domains()
        super(DomainSpecificLDAPandSQLIdentity, self).load_fixtures(fixtures)

    def assert_backends(self):
        _assert_backends(self, assignment='sql', identity={None: 'ldap', 'default': 'ldap', self.domains['domain1']['id']: 'sql'}, resource='sql')

    def config_overrides(self):
        super(DomainSpecificLDAPandSQLIdentity, self).config_overrides()
        self.config_fixture.config(group='resource', driver='sql')
        self.config_fixture.config(group='assignment', driver='sql')
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True, domain_config_dir=unit.TESTCONF + '/domain_configs_one_sql_one_ldap')
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)

    def get_config(self, domain_id):
        return PROVIDERS.identity_api.domain_configs.get_domain_conf(domain_id)

    def test_list_domains(self):
        self.skip_test_overrides('N/A: Not relevant for multi ldap testing')

    def test_delete_domain(self):
        self.assertRaises(exception.DomainNotFound, super(BaseLDAPIdentity, self).test_delete_domain_with_project_api)

    def test_list_users(self):
        _users = self.create_users_across_domains()
        users = PROVIDERS.identity_api.list_users(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
        self.assertEqual(len(default_fixtures.USERS) + 1, len(users))
        user_ids = set((user['id'] for user in users))
        expected_user_ids = set((getattr(self, 'user_%s' % user['name'])['id'] for user in default_fixtures.USERS))
        expected_user_ids.add(_users['user0']['id'])
        for user_ref in users:
            self.assertNotIn('password', user_ref)
        self.assertEqual(expected_user_ids, user_ids)

    def test_domain_segregation(self):
        """Test that separate configs have segregated the domain.

        Test Plan:

        - Users were created in each domain as part of setup, now make sure
          you can only find a given user in its relevant domain/backend
        - Make sure that for a backend that supports multiple domains
          you can get the users via any of its domains

        """
        users = self.create_users_across_domains()
        self.check_user(users['user0'], self.domain_default['id'], http.client.OK)
        self.check_user(users['user0'], self.domains['domain1']['id'], exception.UserNotFound)
        self.check_user(users['user1'], self.domains['domain1']['id'], http.client.OK)
        self.check_user(users['user1'], self.domain_default['id'], exception.UserNotFound)
        self.assertThat(PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id']), matchers.HasLength(1))

    def test_get_domain_mapping_list_is_used(self):
        for i in range(5):
            unit.create_user(PROVIDERS.identity_api, domain_id=self.domains['domain1']['id'])
        with mock.patch.multiple(PROVIDERS.id_mapping_api, get_domain_mapping_list=mock.DEFAULT, get_id_mapping=mock.DEFAULT) as mocked:
            PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id'])
            mocked['get_domain_mapping_list'].assert_called()
            mocked['get_id_mapping'].assert_not_called()

    def test_user_id_comma(self):
        self.skip_test_overrides('Only valid if it is guaranteed to be talking to the fakeldap backend')

    def test_user_enabled_ignored_disable_error(self):
        self.skip_test_overrides("Doesn't apply since LDAP config has no affect on the SQL identity backend.")

    def test_group_enabled_ignored_disable_error(self):
        self.skip_test_overrides("Doesn't apply since LDAP config has no affect on the SQL identity backend.")

    def test_list_role_assignments_filtered_by_role(self):
        base = super(BaseLDAPIdentity, self)
        base.test_list_role_assignments_filtered_by_role()

    def test_delete_domain_with_project_api(self):
        self.assertRaises(exception.DomainNotFound, super(BaseLDAPIdentity, self).test_delete_domain_with_project_api)

    def test_create_project_with_domain_id_and_without_parent_id(self):
        base = super(BaseLDAPIdentity, self)
        base.test_create_project_with_domain_id_and_without_parent_id()

    def test_create_project_with_domain_id_mismatch_to_parent_domain(self):
        base = super(BaseLDAPIdentity, self)
        base.test_create_project_with_domain_id_mismatch_to_parent_domain()

    def test_list_domains_filtered_and_limited(self):
        self.skip_test_overrides('Restricted multi LDAP class does not support multiple domains')

    def test_list_limit_for_domains(self):
        self.skip_test_overrides('Restricted multi LDAP class does not support multiple domains')