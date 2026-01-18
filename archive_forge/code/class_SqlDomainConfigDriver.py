from keystone.common import sql
from keystone.resource.config_backends import sql as config_sql
from keystone.tests import unit
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.resource import test_core
class SqlDomainConfigDriver(unit.BaseTestCase, test_core.DomainConfigDriverTests):

    def setUp(self):
        super(SqlDomainConfigDriver, self).setUp()
        self.useFixture(database.Database())
        self.driver = config_sql.DomainConfig()