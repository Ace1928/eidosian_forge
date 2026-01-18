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
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
@mock.patch.object(ldap.ldapobject.LDAPObject, 'start_tls_s')
def _init_ldap_connection(self, config, mock_ldap_one, mock_ldap_two):
    base_ldap = common_ldap.BaseLdap(config)
    base_ldap.get_connection()