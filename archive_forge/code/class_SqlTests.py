import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
class SqlTests(unit.SQLDriverOverrides, unit.TestCase):

    def setUp(self):
        super(SqlTests, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()
        self.load_fixtures(default_fixtures)
        self.user_foo['enabled'] = True

    def config_files(self):
        config_files = super(SqlTests, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files