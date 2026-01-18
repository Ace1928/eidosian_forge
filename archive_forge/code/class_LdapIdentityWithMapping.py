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
class LdapIdentityWithMapping(BaseLDAPIdentity, unit.SQLDriverOverrides, unit.TestCase):
    """Class to test mapping of default LDAP backend.

    The default configuration is not to enable mapping when using a single
    backend LDAP driver.  However, a cloud provider might want to enable
    the mapping, hence hiding the LDAP IDs from any clients of keystone.
    Setting backward_compatible_ids to False will enable this mapping.

    """

    def config_files(self):
        config_files = super(LdapIdentityWithMapping, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap_sql.conf'))
        return config_files

    def setUp(self):
        super(LdapIdentityWithMapping, self).setUp()
        cache.configure_cache()

    def assert_backends(self):
        _assert_backends(self, identity='ldap')

    def config_overrides(self):
        super(LdapIdentityWithMapping, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)

    def test_dynamic_mapping_build(self):
        """Test to ensure entities not create via controller are mapped.

        Many LDAP backends will, essentially, by Read Only. In these cases
        the mapping is not built by creating objects, rather from enumerating
        the entries.  We test this here my manually deleting the mapping and
        then trying to re-read the entries.

        """
        initial_mappings = len(mapping_sql.list_id_mappings())
        user1 = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user1 = PROVIDERS.identity_api.create_user(user1)
        user2 = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user2 = PROVIDERS.identity_api.create_user(user2)
        mappings = mapping_sql.list_id_mappings()
        self.assertEqual(initial_mappings + 2, len(mappings))
        PROVIDERS.id_mapping_api.purge_mappings({'public_id': user1['id']})
        PROVIDERS.id_mapping_api.purge_mappings({'public_id': user2['id']})
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, user1['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, user2['id'])
        PROVIDERS.identity_api.list_users()
        PROVIDERS.identity_api.get_user(user1['id'])
        PROVIDERS.identity_api.get_user(user2['id'])

    def test_list_domains(self):
        domains = PROVIDERS.resource_api.list_domains()
        default_domain = unit.new_domain_ref(description=u'The default domain', id=CONF.identity.default_domain_id, name=u'Default')
        self.assertEqual([default_domain], domains)