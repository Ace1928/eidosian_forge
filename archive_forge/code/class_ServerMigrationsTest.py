from novaclient import api_versions
from novaclient import base
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_migrations as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_migrations
class ServerMigrationsTest(utils.FixturedTestCase):
    client_fixture_class = client.V1
    data_fixture_class = data.Fixture

    def setUp(self):
        super(ServerMigrationsTest, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.22')

    def test_live_migration_force_complete(self):
        body = {'force_complete': None}
        self.cs.server_migrations.live_migrate_force_complete(1234, 1)
        self.assert_called('POST', '/servers/1234/migrations/1/action', body)