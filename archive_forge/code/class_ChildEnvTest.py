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
class ChildEnvTest(common.HeatTestCase):

    def test_params_flat(self):
        new_params = {'foo': 'bar', 'tester': 'Yes'}
        penv = environment.Environment()
        expected = {'parameters': new_params, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {'resources': {}}}
        cenv = environment.get_child_environment(penv, new_params)
        self.assertEqual(expected, cenv.env_as_dict())

    def test_params_normal(self):
        new_params = {'parameters': {'foo': 'bar', 'tester': 'Yes'}}
        penv = environment.Environment()
        expected = {'parameter_defaults': {}, 'encrypted_param_names': [], 'event_sinks': [], 'resource_registry': {'resources': {}}}
        expected.update(new_params)
        cenv = environment.get_child_environment(penv, new_params)
        self.assertEqual(expected, cenv.env_as_dict())

    def test_params_parent_overwritten(self):
        new_params = {'parameters': {'foo': 'bar', 'tester': 'Yes'}}
        parent_params = {'parameters': {'gone': 'hopefully'}}
        penv = environment.Environment(env=parent_params)
        expected = {'parameter_defaults': {}, 'encrypted_param_names': [], 'event_sinks': [], 'resource_registry': {'resources': {}}}
        expected.update(new_params)
        cenv = environment.get_child_environment(penv, new_params)
        self.assertEqual(expected, cenv.env_as_dict())

    def test_registry_merge_simple(self):
        env1 = {u'resource_registry': {u'OS::Food': u'fruity.yaml'}}
        env2 = {u'resource_registry': {u'OS::Fruit': u'apples.yaml'}}
        penv = environment.Environment(env=env1)
        cenv = environment.get_child_environment(penv, env2)
        rr = cenv.user_env_as_dict()['resource_registry']
        self.assertIn('OS::Food', rr)
        self.assertIn('OS::Fruit', rr)

    def test_registry_merge_favor_child(self):
        env1 = {u'resource_registry': {u'OS::Food': u'carrots.yaml'}}
        env2 = {u'resource_registry': {u'OS::Food': u'apples.yaml'}}
        penv = environment.Environment(env=env1)
        cenv = environment.get_child_environment(penv, env2)
        res = cenv.get_resource_info('OS::Food')
        self.assertEqual('apples.yaml', res.value)

    def test_item_to_remove_simple(self):
        env = {u'resource_registry': {u'OS::Food': u'fruity.yaml'}}
        penv = environment.Environment(env)
        victim = penv.get_resource_info('OS::Food', resource_name='abc')
        self.assertIsNotNone(victim)
        cenv = environment.get_child_environment(penv, None, item_to_remove=victim)
        self.assertRaises(exception.EntityNotFound, cenv.get_resource_info, 'OS::Food', resource_name='abc')
        self.assertNotIn('OS::Food', cenv.user_env_as_dict()['resource_registry'])
        innocent = penv.get_resource_info('OS::Food', resource_name='abc')
        self.assertIsNotNone(innocent)

    def test_item_to_remove_complex(self):
        env = {u'resource_registry': {u'OS::Food': u'fruity.yaml', u'resources': {u'abc': {u'OS::Food': u'nutty.yaml'}}}}
        penv = environment.Environment(env)
        victim = penv.get_resource_info('OS::Food', resource_name='abc')
        self.assertEqual(['resources', 'abc', 'OS::Food'], victim.path)
        cenv = environment.get_child_environment(penv, None, item_to_remove=victim)
        res = cenv.get_resource_info('OS::Food', resource_name='abc')
        self.assertEqual(['OS::Food'], res.path)
        rr = cenv.user_env_as_dict()['resource_registry']
        self.assertIn('OS::Food', rr)
        self.assertNotIn('OS::Food', rr['resources']['abc'])
        innocent2 = penv.get_resource_info('OS::Food', resource_name='abc')
        self.assertEqual(['resources', 'abc', 'OS::Food'], innocent2.path)

    def test_item_to_remove_none(self):
        env = {u'resource_registry': {u'OS::Food': u'fruity.yaml'}}
        penv = environment.Environment(env)
        victim = penv.get_resource_info('OS::Food', resource_name='abc')
        self.assertIsNotNone(victim)
        cenv = environment.get_child_environment(penv, None)
        res = cenv.get_resource_info('OS::Food', resource_name='abc')
        self.assertIsNotNone(res)

    def test_drill_down_to_child_resource(self):
        env = {u'resource_registry': {u'OS::Food': u'fruity.yaml', u'resources': {u'a': {u'OS::Fruit': u'apples.yaml', u'hooks': 'pre-create'}, u'nested': {u'b': {u'OS::Fruit': u'carrots.yaml'}, u'nested_res': {u'hooks': 'pre-create'}}}}}
        penv = environment.Environment(env)
        cenv = environment.get_child_environment(penv, None, child_resource_name=u'nested')
        registry = cenv.user_env_as_dict()['resource_registry']
        resources = registry['resources']
        self.assertIn('nested_res', resources)
        self.assertIn('hooks', resources['nested_res'])
        self.assertIsNotNone(cenv.get_resource_info('OS::Food', resource_name='abc'))
        self.assertRaises(exception.EntityNotFound, cenv.get_resource_info, 'OS::Fruit', resource_name='a')
        res = cenv.get_resource_info('OS::Fruit', resource_name='b')
        self.assertIsNotNone(res)
        self.assertEqual(u'carrots.yaml', res.value)

    def test_drill_down_non_matching_wildcard(self):
        env = {u'resource_registry': {u'resources': {u'nested': {u'c': {u'OS::Fruit': u'carrots.yaml', u'hooks': 'pre-create'}}, u'*_doesnt_match_nested': {u'nested_res': {u'hooks': 'pre-create'}}}}}
        penv = environment.Environment(env)
        cenv = environment.get_child_environment(penv, None, child_resource_name=u'nested')
        registry = cenv.user_env_as_dict()['resource_registry']
        resources = registry['resources']
        self.assertIn('c', resources)
        self.assertNotIn('nested_res', resources)
        res = cenv.get_resource_info('OS::Fruit', resource_name='c')
        self.assertIsNotNone(res)
        self.assertEqual(u'carrots.yaml', res.value)

    def test_drill_down_matching_wildcard(self):
        env = {u'resource_registry': {u'resources': {u'nested': {u'c': {u'OS::Fruit': u'carrots.yaml', u'hooks': 'pre-create'}}, u'nest*': {u'nested_res': {u'hooks': 'pre-create'}}}}}
        penv = environment.Environment(env)
        cenv = environment.get_child_environment(penv, None, child_resource_name=u'nested')
        registry = cenv.user_env_as_dict()['resource_registry']
        resources = registry['resources']
        self.assertIn('c', resources)
        self.assertIn('nested_res', resources)
        res = cenv.get_resource_info('OS::Fruit', resource_name='c')
        self.assertIsNotNone(res)
        self.assertEqual(u'carrots.yaml', res.value)

    def test_drill_down_prefer_exact_match(self):
        env = {u'resource_registry': {u'resources': {u'*esource': {u'hooks': 'pre-create'}, u'res*': {u'hooks': 'pre-create'}, u'resource': {u'OS::Fruit': u'carrots.yaml', u'hooks': 'pre-update'}, u'resource*': {u'hooks': 'pre-create'}, u'*resource': {u'hooks': 'pre-create'}, u'*sour*': {u'hooks': 'pre-create'}}}}
        penv = environment.Environment(env)
        cenv = environment.get_child_environment(penv, None, child_resource_name=u'resource')
        registry = cenv.user_env_as_dict()['resource_registry']
        resources = registry['resources']
        self.assertEqual(u'carrots.yaml', resources[u'OS::Fruit'])
        self.assertEqual('pre-update', resources[u'hooks'])