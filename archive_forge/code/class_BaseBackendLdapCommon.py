import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class BaseBackendLdapCommon(object):
    """Mixin class to set up generic LDAP backends."""

    def setUp(self):
        super(BaseBackendLdapCommon, self).setUp()
        self.useFixture(ldapdb.LDAPDatabase())
        self.load_backends()
        self.load_fixtures(default_fixtures)

    def _get_domain_fixture(self):
        """Return the static domain, since domains in LDAP are read-only."""
        return PROVIDERS.resource_api.get_domain(CONF.identity.default_domain_id)

    def get_config(self, domain_id):
        return CONF

    def config_overrides(self):
        super(BaseBackendLdapCommon, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')

    def config_files(self):
        config_files = super(BaseBackendLdapCommon, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap.conf'))
        return config_files

    def get_user_enabled_vals(self, user):
        user_dn = PROVIDERS.identity_api.driver.user._id_to_dn_string(user['id'])
        enabled_attr_name = CONF.ldap.user_enabled_attribute
        ldap_ = PROVIDERS.identity_api.driver.user.get_connection()
        res = ldap_.search_s(user_dn, ldap.SCOPE_BASE, u'(sn=%s)' % user['name'])
        if enabled_attr_name in res[0][1]:
            return res[0][1][enabled_attr_name]
        else:
            return None