import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class EnvironmentTest(common.HeatTestCase):

    def setUp(self):
        super(EnvironmentTest, self).setUp()
        self.g_env = resources.global_env()

    def test_load_old_parameters(self):
        old = {u'a': u'ff', u'b': u'ss'}
        expected = {u'parameters': old, u'encrypted_param_names': [], u'parameter_defaults': {}, u'event_sinks': [], u'resource_registry': {u'resources': {}}}
        env = environment.Environment(old)
        self.assertEqual(expected, env.env_as_dict())
        del expected['encrypted_param_names']
        self.assertEqual(expected, env.user_env_as_dict())

    def test_load_new_env(self):
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'encrypted_param_names': [], u'parameter_defaults': {u'ff': 'new_def'}, u'event_sinks': [], u'resource_registry': {u'OS::Food': u'fruity.yaml', u'resources': {}}}
        env = environment.Environment(new_env)
        self.assertEqual(new_env, env.env_as_dict())
        del new_env['encrypted_param_names']
        self.assertEqual(new_env, env.user_env_as_dict())

    def test_global_registry(self):
        self.g_env.register_class('CloudX::Nova::Server', generic_resource.GenericResource)
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::*': 'CloudX::*'}}
        env = environment.Environment(new_env)
        self.assertEqual('CloudX::Nova::Server', env.get_resource_info('OS::Nova::Server', 'my_db_server').name)

    def test_global_registry_many_to_one(self):
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::Nova::*': 'OS::Heat::None'}}
        env = environment.Environment(new_env)
        self.assertEqual('OS::Heat::None', env.get_resource_info('OS::Nova::Server', 'my_db_server').name)

    def test_global_registry_many_to_one_no_recurse(self):
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::*': 'OS::Heat::None'}}
        env = environment.Environment(new_env)
        self.assertEqual('OS::Heat::None', env.get_resource_info('OS::Some::Name', 'my_db_server').name)

    def test_map_one_resource_type(self):
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'resources': {u'my_db_server': {u'OS::DBInstance': 'db.yaml'}}}}
        env = environment.Environment(new_env)
        info = env.get_resource_info('OS::DBInstance', 'my_db_server')
        self.assertEqual('db.yaml', info.value)

    def test_map_all_resources_of_type(self):
        self.g_env.register_class('OS::Nova::FloatingIP', generic_resource.GenericResource)
        new_env = {u'parameters': {u'a': u'ff', u'b': u'ss'}, u'resource_registry': {u'OS::Networking::FloatingIP': 'OS::Nova::FloatingIP', u'OS::Loadbalancer': 'lb.yaml'}}
        env = environment.Environment(new_env)
        self.assertEqual('OS::Nova::FloatingIP', env.get_resource_info('OS::Networking::FloatingIP', 'my_fip').name)

    def test_resource_sort_order_len(self):
        new_env = {u'resource_registry': {u'resources': {u'my_fip': {u'OS::Networking::FloatingIP': 'ip.yaml'}}}, u'OS::Networking::FloatingIP': 'OS::Nova::FloatingIP'}
        env = environment.Environment(new_env)
        self.assertEqual('ip.yaml', env.get_resource_info('OS::Networking::FloatingIP', 'my_fip').value)

    def test_env_load(self):
        new_env = {u'resource_registry': {u'resources': {u'my_fip': {u'OS::Networking::FloatingIP': 'ip.yaml'}}}}
        env = environment.Environment()
        self.assertRaises(exception.EntityNotFound, env.get_resource_info, 'OS::Networking::FloatingIP', 'my_fip')
        env.load(new_env)
        self.assertEqual('ip.yaml', env.get_resource_info('OS::Networking::FloatingIP', 'my_fip').value)

    def test_register_with_path(self):
        yaml_env = '\n        resource_registry:\n          test::one: a.yaml\n          resources:\n            res_x:\n              test::two: b.yaml\n'
        env = environment.Environment(environment_format.parse(yaml_env))
        self.assertEqual('a.yaml', env.get_resource_info('test::one').value)
        self.assertEqual('b.yaml', env.get_resource_info('test::two', 'res_x').value)
        env2 = environment.Environment()
        env2.register_class('test::one', 'a.yaml', path=['test::one'])
        env2.register_class('test::two', 'b.yaml', path=['resources', 'res_x', 'test::two'])
        self.assertEqual(env.env_as_dict(), env2.env_as_dict())

    def test_constraints(self):
        env = environment.Environment({})
        first_constraint = object()
        second_constraint = object()
        env.register_constraint('constraint1', first_constraint)
        env.register_constraint('constraint2', second_constraint)
        self.assertIs(first_constraint, env.get_constraint('constraint1'))
        self.assertIs(second_constraint, env.get_constraint('constraint2'))
        self.assertIsNone(env.get_constraint('no_constraint'))

    def test_constraints_registry(self):
        constraint_content = '\nclass MyConstraint(object):\n    pass\n\ndef constraint_mapping():\n    return {"constraint1": MyConstraint}\n        '
        plugin_dir = self.useFixture(fixtures.TempDir())
        plugin_file = os.path.join(plugin_dir.path, 'test.py')
        with open(plugin_file, 'w+') as ef:
            ef.write(constraint_content)
        cfg.CONF.set_override('plugin_dirs', plugin_dir.path)
        env = environment.Environment({})
        resources._load_global_environment(env)
        self.assertEqual('MyConstraint', env.get_constraint('constraint1').__name__)
        self.assertIsNone(env.get_constraint('no_constraint'))

    def test_constraints_registry_error(self):
        constraint_content = '\ndef constraint_mapping():\n    raise ValueError("oops")\n        '
        plugin_dir = self.useFixture(fixtures.TempDir())
        plugin_file = os.path.join(plugin_dir.path, 'test.py')
        with open(plugin_file, 'w+') as ef:
            ef.write(constraint_content)
        cfg.CONF.set_override('plugin_dirs', plugin_dir.path)
        env = environment.Environment({})
        error = self.assertRaises(ValueError, resources._load_global_environment, env)
        self.assertEqual('oops', str(error))

    def test_constraints_registry_stevedore(self):
        env = environment.Environment({})
        resources._load_global_environment(env)
        self.assertEqual('FlavorConstraint', env.get_constraint('nova.flavor').__name__)
        self.assertIsNone(env.get_constraint('no_constraint'))

    def test_event_sinks(self):
        env = environment.Environment({'event_sinks': [{'type': 'zaqar-queue', 'target': 'myqueue'}]})
        self.assertEqual([{'type': 'zaqar-queue', 'target': 'myqueue'}], env.user_env_as_dict()['event_sinks'])
        sinks = env.get_event_sinks()
        self.assertEqual(1, len(sinks))
        self.assertEqual('myqueue', sinks[0]._target)

    def test_event_sinks_load(self):
        env = environment.Environment()
        self.assertEqual([], env.get_event_sinks())
        env.load({'event_sinks': [{'type': 'zaqar-queue', 'target': 'myqueue'}]})
        self.assertEqual([{'type': 'zaqar-queue', 'target': 'myqueue'}], env.user_env_as_dict()['event_sinks'])