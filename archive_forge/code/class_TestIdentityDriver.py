from oslo_config import fixture as config_fixture
from keystone.identity.backends import ldap
from keystone.tests.unit import core
from keystone.tests.unit.identity.backends import test_base
from keystone.tests.unit.ksfixtures import ldapdb
class TestIdentityDriver(core.BaseTestCase, test_base.IdentityDriverTests):
    allows_name_update = False
    allows_self_service_change_password = False
    expected_is_domain_aware = False
    expected_default_assignment_driver = 'sql'
    expected_is_sql = False
    expected_generates_uuids = False

    def setUp(self):
        super(TestIdentityDriver, self).setUp()
        config_fixture_ = self.useFixture(config_fixture.Config())
        config_fixture_.config(group='ldap', url='fake://memory', user='cn=Admin', password='password', suffix='cn=example,cn=com')
        self.useFixture(ldapdb.LDAPDatabase())
        self.driver = ldap.Identity()

    def test_delete_user(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_delete_user_no_user_exc(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_delete_group(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_delete_group_doesnt_exist_exc(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_remove_user_from_group(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_remove_user_from_group_not_in_group(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_remove_user_from_group_no_user(self):
        self.skip_test_overrides('N/A: LDAP has no write support')

    def test_remove_user_from_group_no_group(self):
        self.skip_test_overrides('N/A: LDAP has no write support')