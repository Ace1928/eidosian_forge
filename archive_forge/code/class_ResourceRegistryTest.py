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
class ResourceRegistryTest(common.HeatTestCase):

    def test_resources_load(self):
        resources = {u'pre_create': {u'OS::Fruit': u'apples.yaml', u'hooks': 'pre-create'}, u'pre_update': {u'hooks': 'pre-update'}, u'both': {u'hooks': ['pre-create', 'pre-update']}, u'b': {u'OS::Food': u'fruity.yaml'}, u'nested': {u'res': {u'hooks': 'pre-create'}}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({'resources': resources})
        self.assertIsNotNone(registry.get_resource_info('OS::Fruit', resource_name='pre_create'))
        self.assertIsNotNone(registry.get_resource_info('OS::Food', resource_name='b'))
        resources = registry.as_dict()['resources']
        self.assertEqual('pre-create', resources['pre_create']['hooks'])
        self.assertEqual('pre-update', resources['pre_update']['hooks'])
        self.assertEqual(['pre-create', 'pre-update'], resources['both']['hooks'])
        self.assertEqual('pre-create', resources['nested']['res']['hooks'])

    def test_load_registry_invalid_hook_type(self):
        resources = {u'resources': {u'a': {u'hooks': 'invalid-type'}}}
        registry = environment.ResourceRegistry(None, {})
        msg = 'Invalid hook type "invalid-type" for resource breakpoint, acceptable hook types are: (\'pre-create\', \'pre-update\', \'pre-delete\', \'post-create\', \'post-update\', \'post-delete\')'
        ex = self.assertRaises(exception.InvalidBreakPointHook, registry.load, {'resources': resources})
        self.assertEqual(msg, str(ex))

    def test_list_type_validation_invalid_support_status(self):
        registry = environment.ResourceRegistry(None, {})
        ex = self.assertRaises(exception.Invalid, registry.get_types, support_status='junk')
        msg = 'Invalid support status and should be one of %s' % str(support.SUPPORT_STATUSES)
        self.assertIn(msg, ex.message)

    def test_list_type_validation_valid_support_status(self):
        registry = environment.ResourceRegistry(None, {})
        for status in support.SUPPORT_STATUSES:
            self.assertEqual([], registry.get_types(support_status=status))

    def test_list_type_find_by_status(self):
        registry = resources.global_env().registry
        types = registry.get_types(support_status=support.UNSUPPORTED)
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertNotIn('GenericResourceType', types)

    def test_list_type_find_by_status_none(self):
        registry = resources.global_env().registry
        types = registry.get_types(support_status=None)
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertIn('GenericResourceType', types)

    def test_list_type_with_name(self):
        registry = resources.global_env().registry
        types = registry.get_types(type_name='ResourceType*')
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertNotIn('GenericResourceType', types)

    def test_list_type_with_name_none(self):
        registry = resources.global_env().registry
        types = registry.get_types(type_name=None)
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertIn('GenericResourceType', types)

    def test_list_type_with_is_available_exception(self):
        registry = resources.global_env().registry
        self.patchobject(generic_resource.GenericResource, 'is_service_available', side_effect=exception.ClientNotAvailable(client_name='generic'))
        types = registry.get_types(utils.dummy_context(), type_name='GenericResourceType')
        self.assertNotIn('GenericResourceType', types)

    def test_list_type_with_invalid_type_name(self):
        registry = resources.global_env().registry
        types = registry.get_types(type_name="r'[^\\+]'")
        self.assertEqual([], types)

    def test_list_type_with_version(self):
        registry = resources.global_env().registry
        types = registry.get_types(version='5.0.0')
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertNotIn('ResourceTypeSupportedKilo', types)

    def test_list_type_with_version_none(self):
        registry = resources.global_env().registry
        types = registry.get_types(version=None)
        self.assertIn('ResourceTypeUnSupportedLiberty', types)
        self.assertIn('ResourceTypeSupportedKilo', types)

    def test_list_type_with_version_invalid(self):
        registry = resources.global_env().registry
        types = registry.get_types(version='invalid')
        self.assertEqual([], types)