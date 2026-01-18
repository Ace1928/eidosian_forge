from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
class MigrationsV280Test(MigrationsV266Test):

    def setUp(self):
        super(MigrationsV280Test, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.80')

    def test_list_migrations_with_user_id(self):
        user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
        params = {'user_id': user_id}
        ms = self.cs.migrations.list(**params)
        self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-migrations?user_id=%s' % user_id)
        for m in ms:
            self.assertIsInstance(m, migrations.Migration)

    def test_list_migrations_with_project_id(self):
        project_id = 'b59c18e5fa284fd384987c5cb25a1853'
        params = {'project_id': project_id}
        ms = self.cs.migrations.list(**params)
        self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-migrations?project_id=%s' % project_id)
        for m in ms:
            self.assertIsInstance(m, migrations.Migration)

    def test_list_migrations_with_user_and_project_id(self):
        user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
        project_id = 'b59c18e5fa284fd384987c5cb25a1853'
        params = {'user_id': user_id, 'project_id': project_id}
        ms = self.cs.migrations.list(**params)
        self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-migrations?project_id=%s&user_id=%s' % (project_id, user_id))
        for m in ms:
            self.assertIsInstance(m, migrations.Migration)

    def test_list_migrations_with_user_id_pre_v280(self):
        self.cs.api_version = api_versions.APIVersion('2.79')
        user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
        ex = self.assertRaises(TypeError, self.cs.migrations.list, user_id=user_id)
        self.assertIn("unexpected keyword argument 'user_id'", str(ex))

    def test_list_migrations_with_project_id_pre_v280(self):
        self.cs.api_version = api_versions.APIVersion('2.79')
        project_id = '23cc0930d27c4be0acc14d7c47a3e1f7'
        ex = self.assertRaises(TypeError, self.cs.migrations.list, project_id=project_id)
        self.assertIn("unexpected keyword argument 'project_id'", str(ex))