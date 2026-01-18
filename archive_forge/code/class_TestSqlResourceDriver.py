from keystone.resource.backends import sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.resource import test_backends
class TestSqlResourceDriver(unit.BaseTestCase, test_backends.ResourceDriverTests):

    def setUp(self):
        super(TestSqlResourceDriver, self).setUp()
        self.useFixture(database.Database())
        self.driver = sql.Resource()
        root_domain = default_fixtures.ROOT_DOMAIN
        root_domain['domain_id'] = root_domain['id']
        root_domain['is_domain'] = True
        self.driver.create_project(root_domain['id'], root_domain)