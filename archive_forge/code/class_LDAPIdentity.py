from unittest import mock
import fixtures
import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap
class LDAPIdentity(LdapPoolCommonTestMixin, test_backend_ldap.LDAPIdentity, unit.TestCase):
    """Executes tests in existing base class with pooled LDAP handler."""

    def setUp(self):
        self.useFixture(fixtures.MockPatchObject(common_ldap.PooledLDAPHandler, 'Connector', fakeldap.FakeLdapPool))
        super(LDAPIdentity, self).setUp()
        self.addCleanup(self.cleanup_pools)
        self.conn_pools = common_ldap.PooledLDAPHandler.connection_pools
        PROVIDERS.identity_api.get_user(self.user_foo['id'])

    def config_files(self):
        config_files = super(LDAPIdentity, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap_pool.conf'))
        return config_files