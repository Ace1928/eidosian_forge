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
class DomainSpecificSQLIdentity(DomainSpecificLDAPandSQLIdentity):
    """Class to test simplest use of domain-specific SQL driver.

    The simplest use of an SQL domain-specific backend is when it is used to
    augment the standard case when LDAP is the default driver defined in the
    main config file. This would allow, for example, service users to be
    stored in SQL while LDAP handles the rest. Hence we define:

    - The default driver uses the LDAP backend for the default domain
    - A separate SQL backend for domain1

    """
    DOMAIN_COUNT = 2
    DOMAIN_SPECIFIC_COUNT = 1

    def assert_backends(self):
        _assert_backends(self, assignment='sql', identity='ldap', resource='sql')

    def config_overrides(self):
        super(DomainSpecificSQLIdentity, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True, domain_config_dir=unit.TESTCONF + '/domain_configs_default_ldap_one_sql')
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=True)

    def get_config(self, domain_id):
        if domain_id == CONF.identity.default_domain_id:
            return CONF
        else:
            return PROVIDERS.identity_api.domain_configs.get_domain_conf(domain_id)

    def test_default_sql_plus_sql_specific_driver_fails(self):
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='assignment', driver='sql')
        self.load_backends()
        PROVIDERS.identity_api.list_users(domain_scope=CONF.identity.default_domain_id)
        self.assertIsNotNone(self.get_config(self.domains['domain1']['id']))
        self.config_fixture.config(group='identity', driver='sql')
        self.config_fixture.config(group='assignment', driver='sql')
        self.load_backends()
        self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.list_users, domain_scope=CONF.identity.default_domain_id)

    def test_multiple_sql_specific_drivers_fails(self):
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='assignment', driver='sql')
        self.load_backends()
        self.domain_count = 3
        self.setup_initial_domains()
        PROVIDERS.identity_api.list_users(domain_scope=CONF.identity.default_domain_id)
        self.assertIsNotNone(self.get_config(self.domains['domain1']['id']))
        self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs._load_config_from_file, PROVIDERS.resource_api, [unit.TESTCONF + '/domain_configs_one_extra_sql/' + 'keystone.domain2.conf'], 'domain2')