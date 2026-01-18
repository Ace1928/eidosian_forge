from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
class MigrationsV259Test(MigrationsV223Test):

    def setUp(self):
        super(MigrationsV259Test, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.59')

    def test_list_migrations(self):
        ml = self.cs.migrations.list()
        self.assert_request_id(ml, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-migrations')
        for m in ml:
            self.assertIsInstance(m, migrations.Migration)
            self.assertEqual(m.migration_type, 'live-migration')
            self.assertTrue(hasattr(m, 'uuid'))

    def test_list_migrations_with_limit_marker_params(self):
        marker = '12140183-c814-4ddf-8453-6df43028aa67'
        params = {'limit': 10, 'marker': marker, 'changes_since': '2012-02-29T06:23:22'}
        ms = self.cs.migrations.list(**params)
        self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-migrations?changes-since=%s&limit=10&marker=%s' % ('2012-02-29T06%3A23%3A22', marker))
        for m in ms:
            self.assertIsInstance(m, migrations.Migration)