import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class BaseBackendLdapIdentitySqlEverythingElse(unit.SQLDriverOverrides):
    """Mixin base for Identity LDAP, everything else SQL backend tests."""

    def config_files(self):
        config_files = super(BaseBackendLdapIdentitySqlEverythingElse, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap_sql.conf'))
        return config_files

    def setUp(self):
        sqldb = self.useFixture(database.Database())
        super(BaseBackendLdapIdentitySqlEverythingElse, self).setUp()
        self.load_backends()
        cache.configure_cache()
        sqldb.recreate()
        self.load_fixtures(default_fixtures)
        self.user_foo['enabled'] = True

    def config_overrides(self):
        super(BaseBackendLdapIdentitySqlEverythingElse, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='resource', driver='sql')
        self.config_fixture.config(group='assignment', driver='sql')