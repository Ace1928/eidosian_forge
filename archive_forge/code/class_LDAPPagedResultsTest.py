import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class LDAPPagedResultsTest(unit.TestCase):
    """Test the paged results functionality in keystone.common.ldap.core."""

    def setUp(self):
        super(LDAPPagedResultsTest, self).setUp()
        self.useFixture(ldapdb.LDAPDatabase())
        self.useFixture(database.Database())
        self.load_backends()
        self.load_fixtures(default_fixtures)

    def config_overrides(self):
        super(LDAPPagedResultsTest, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')

    def config_files(self):
        config_files = super(LDAPPagedResultsTest, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap.conf'))
        return config_files

    @mock.patch.object(fakeldap.FakeLdap, 'search_ext')
    @mock.patch.object(fakeldap.FakeLdap, 'result3')
    def test_paged_results_control_api(self, mock_result3, mock_search_ext):
        mock_result3.return_value = ('', [], 1, [])
        self.config_fixture.config(group='ldap', page_size=1)
        conn = PROVIDERS.identity_api.user.get_connection()
        conn._paged_search_s('dc=example,dc=test', ldap.SCOPE_SUBTREE, 'objectclass=*', ['mail', 'userPassword'])
        args, _ = mock_search_ext.call_args
        self.assertEqual(('dc=example,dc=test', 2, 'objectclass=*'), args[0:3])
        attrlist = sorted([attr for attr in args[3] if attr])
        self.assertEqual(['mail', 'userPassword'], attrlist)