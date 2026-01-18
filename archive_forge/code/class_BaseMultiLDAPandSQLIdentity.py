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
class BaseMultiLDAPandSQLIdentity(object):
    """Mixin class with support methods for domain-specific config testing."""

    def create_users_across_domains(self):
        """Create a set of users, each with a role on their own domain."""
        initial_mappings = len(mapping_sql.list_id_mappings())
        users = {}
        users['user0'] = unit.create_user(PROVIDERS.identity_api, self.domain_default['id'])
        PROVIDERS.assignment_api.create_grant(user_id=users['user0']['id'], domain_id=self.domain_default['id'], role_id=self.role_member['id'])
        for x in range(1, self.domain_count):
            users['user%s' % x] = unit.create_user(PROVIDERS.identity_api, self.domains['domain%s' % x]['id'])
            PROVIDERS.assignment_api.create_grant(user_id=users['user%s' % x]['id'], domain_id=self.domains['domain%s' % x]['id'], role_id=self.role_member['id'])
        self.assertEqual(initial_mappings + self.domain_specific_count, len(mapping_sql.list_id_mappings()))
        return users

    def check_user(self, user, domain_id, expected_status):
        """Check user is in correct backend.

        As part of the tests, we want to force ourselves to manually
        select the driver for a given domain, to make sure the entity
        ended up in the correct backend.

        """
        driver = PROVIDERS.identity_api._select_identity_driver(domain_id)
        unused, unused, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(user['id'])
        if expected_status == http.client.OK:
            ref = driver.get_user(entity_id)
            ref = PROVIDERS.identity_api._set_domain_id_and_mapping(ref, domain_id, driver, map.EntityType.USER)
            user = user.copy()
            del user['password']
            self.assertDictEqual(user, ref)
        else:
            try:
                driver.get_user(entity_id)
            except expected_status:
                pass

    def setup_initial_domains(self):

        def create_domain(domain):
            try:
                ref = PROVIDERS.resource_api.create_domain(domain['id'], domain)
            except exception.Conflict:
                ref = PROVIDERS.resource_api.get_domain_by_name(domain['name'])
            return ref
        self.domains = {}
        for x in range(1, self.domain_count):
            domain = 'domain%s' % x
            self.domains[domain] = create_domain({'id': uuid.uuid4().hex, 'name': domain})

    def test_authenticate_to_each_domain(self):
        """Test that a user in each domain can authenticate."""
        users = self.create_users_across_domains()
        for user_num in range(self.domain_count):
            user = 'user%s' % user_num
            with self.make_request():
                PROVIDERS.identity_api.authenticate(user_id=users[user]['id'], password=users[user]['password'])