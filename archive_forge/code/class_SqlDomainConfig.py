from keystone.common import sql
from keystone.resource.config_backends import sql as config_sql
from keystone.tests import unit
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.resource import test_core
class SqlDomainConfig(core_sql.BaseBackendSqlTests, test_core.DomainConfigTests):

    def setUp(self):
        super(SqlDomainConfig, self).setUp()
        test_core.DomainConfigTests.setUp(self)