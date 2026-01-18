import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class EnvironmentCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with environments."""

    def setUp(self):
        super(EnvironmentCLITests, self).setUp()
        self.create_file('env.yaml', 'name: env\ndescription: Test env\nvariables:\n  var: "value"')

    def test_environment_create(self):
        env = self.mistral_admin('environment-create', params='env.yaml')
        env_name = self.get_field_value(env, 'Name')
        env_desc = self.get_field_value(env, 'Description')
        self.assertTableStruct(env, ['Field', 'Value'])
        envs = self.mistral_admin('environment-list')
        self.assertIn(env_name, [en['Name'] for en in envs])
        self.assertIn(env_desc, [en['Description'] for en in envs])
        self.mistral_admin('environment-delete', params=env_name)
        envs = self.mistral_admin('environment-list')
        self.assertNotIn(env_name, [en['Name'] for en in envs])

    def test_environment_create_without_description(self):
        self.create_file('env_without_des.yaml', 'name: env\nvariables:\n  var: "value"')
        env = self.mistral_admin('environment-create', params='env_without_des.yaml')
        env_name = self.get_field_value(env, 'Name')
        env_desc = self.get_field_value(env, 'Description')
        self.assertTableStruct(env, ['Field', 'Value'])
        envs = self.mistral_admin('environment-list')
        self.assertIn(env_name, [en['Name'] for en in envs])
        self.assertIn(env_desc, 'None')
        self.mistral_admin('environment-delete', params='env')
        envs = self.mistral_admin('environment-list')
        self.assertNotIn(env_name, [en['Name'] for en in envs])

    def test_environment_update(self):
        env = self.environment_create('env.yaml')
        env_name = self.get_field_value(env, 'Name')
        env_desc = self.get_field_value(env, 'Description')
        env_created_at = self.get_field_value(env, 'Created at')
        env_updated_at = self.get_field_value(env, 'Updated at')
        self.assertIsNotNone(env_created_at)
        self.assertEqual('None', env_updated_at)
        self.create_file('env_upd.yaml', 'name: env\ndescription: Updated env\nvariables:\n  var: "value"')
        env = self.mistral_admin('environment-update', params='env_upd.yaml')
        self.assertTableStruct(env, ['Field', 'Value'])
        updated_env_name = self.get_field_value(env, 'Name')
        updated_env_desc = self.get_field_value(env, 'Description')
        updated_env_created_at = self.get_field_value(env, 'Created at')
        updated_env_updated_at = self.get_field_value(env, 'Updated at')
        self.assertEqual(env_name, updated_env_name)
        self.assertNotEqual(env_desc, updated_env_desc)
        self.assertEqual('Updated env', updated_env_desc)
        self.assertEqual(env_created_at.split('.')[0], updated_env_created_at)
        self.assertIsNotNone(updated_env_updated_at)

    def test_environment_get(self):
        env = self.environment_create('env.yaml')
        env_name = self.get_field_value(env, 'Name')
        env_desc = self.get_field_value(env, 'Description')
        env = self.mistral_admin('environment-get', params=env_name)
        fetched_env_name = self.get_field_value(env, 'Name')
        fetched_env_desc = self.get_field_value(env, 'Description')
        self.assertTableStruct(env, ['Field', 'Value'])
        self.assertEqual(env_name, fetched_env_name)
        self.assertEqual(env_desc, fetched_env_desc)

    def test_environment_get_export(self):
        env = self.environment_create('env.yaml')
        env_name = self.get_field_value(env, 'Name')
        env_desc = self.get_field_value(env, 'Description')
        env = self.mistral_admin('environment-get', params='--export {0}'.format(env_name))
        fetched_env_name = self.get_field_value(env, 'name')
        fetched_env_desc = self.get_field_value(env, 'description')
        self.assertTableStruct(env, ['Field', 'Value'])
        self.assertEqual(env_name, fetched_env_name)
        self.assertEqual(env_desc, fetched_env_desc)