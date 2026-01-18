import fixtures
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests.unit import fakeldap
class LDAPDatabase(fixtures.Fixture):
    """A fixture for setting up and tearing down an LDAP database."""

    def __init__(self, dbclass=fakeldap.FakeLdap):
        self._dbclass = dbclass

    def setUp(self):
        super(LDAPDatabase, self).setUp()
        self.clear()
        common_ldap.WRITABLE = True
        common_ldap._HANDLERS.clear()
        common_ldap.register_handler('fake://', self._dbclass)
        self.addCleanup(self.clear)
        self.addCleanup(common_ldap._HANDLERS.clear)
        self.addCleanup(self.disable_write)

    def disable_write(self):
        common_ldap.WRITABLE = False

    def clear(self):
        for shelf in fakeldap.FakeShelves:
            fakeldap.FakeShelves[shelf].clear()