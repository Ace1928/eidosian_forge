import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
Return the static domain, since domains in LDAP are read-only.