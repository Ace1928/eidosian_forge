from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
class EnvironmentIsolationCLITests(base_v2.MistralClientTestBase):

    def setUp(self):
        super(EnvironmentIsolationCLITests, self).setUp()
        self.env_file = 'env.yaml'
        self.create_file('{0}'.format(self.env_file), 'name: env\ndescription: Test env\nvariables:\n  var: value')

    def test_environment_name_uniqueness(self):
        self.environment_create(self.env_file)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-create', params=self.env_file)
        self.environment_create(self.env_file, admin=False)
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'environment-create', params=self.env_file)

    def test_environment_isolation(self):
        env = self.environment_create(self.env_file)
        env_name = self.get_field_value(env, 'Name')
        envs = self.mistral_admin('environment-list')
        self.assertIn(env_name, [en['Name'] for en in envs])
        alt_envs = self.mistral_alt_user('environment-list')
        self.assertNotIn(env_name, [en['Name'] for en in alt_envs])

    def test_get_env_from_another_tenant(self):
        env = self.environment_create(self.env_file)
        env_name = self.get_field_value(env, 'Name')
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'environment-get', params=env_name)

    def test_delete_env_from_another_tenant(self):
        env = self.environment_create(self.env_file)
        env_name = self.get_field_value(env, 'Name')
        self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'environment-delete', params=env_name)