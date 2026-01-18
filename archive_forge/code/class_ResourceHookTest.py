import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
class ResourceHookTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceHookTest, self).setUp()
        self.env = environment.Environment()
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=self.env), stack_id=str(uuid.uuid4()))

    def test_hook(self):
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.data = mock.Mock(return_value={})
        self.assertFalse(res.has_hook('pre-create'))
        self.assertFalse(res.has_hook('pre-update'))
        res.data = mock.Mock(return_value={'pre-create': 'True'})
        self.assertTrue(res.has_hook('pre-create'))
        self.assertFalse(res.has_hook('pre-update'))
        res.data = mock.Mock(return_value={'pre-create': 'False'})
        self.assertFalse(res.has_hook('pre-create'))
        self.assertFalse(res.has_hook('pre-update'))
        res.data = mock.Mock(return_value={'pre-update': 'True'})
        self.assertFalse(res.has_hook('pre-create'))
        self.assertTrue(res.has_hook('pre-update'))
        res.data = mock.Mock(return_value={'pre-delete': 'True'})
        self.assertFalse(res.has_hook('pre-create'))
        self.assertFalse(res.has_hook('pre-update'))
        self.assertTrue(res.has_hook('pre-delete'))
        res.data = mock.Mock(return_value={'post-create': 'True'})
        self.assertFalse(res.has_hook('post-delete'))
        self.assertFalse(res.has_hook('post-update'))
        self.assertTrue(res.has_hook('post-create'))
        res.data = mock.Mock(return_value={'post-update': 'True'})
        self.assertFalse(res.has_hook('post-create'))
        self.assertFalse(res.has_hook('post-delete'))
        self.assertTrue(res.has_hook('post-update'))
        res.data = mock.Mock(return_value={'post-delete': 'True'})
        self.assertFalse(res.has_hook('post-create'))
        self.assertFalse(res.has_hook('post-update'))
        self.assertTrue(res.has_hook('post-delete'))

    def test_set_hook(self):
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.data_set = mock.Mock()
        res.data_delete = mock.Mock()
        res.trigger_hook('pre-create')
        res.data_set.assert_called_with('pre-create', 'True')
        res.trigger_hook('pre-update')
        res.data_set.assert_called_with('pre-update', 'True')
        res.clear_hook('pre-create')
        res.data_delete.assert_called_with('pre-create')

    def test_signal_clear_hook(self):
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.clear_hook = mock.Mock()
        res.has_hook = mock.Mock(return_value=True)
        self.assertRaises(exception.ResourceActionNotSupported, res.signal, None)
        self.assertFalse(res.clear_hook.called)
        self.assertRaises(exception.ResourceActionNotSupported, res.signal, {'other_hook': 'alarm'})
        self.assertFalse(res.clear_hook.called)
        self.assertRaises(exception.InvalidBreakPointHook, res.signal, {'unset_hook': 'unknown_hook'})
        self.assertFalse(res.clear_hook.called)
        result = res.signal({'unset_hook': 'pre-create'})
        res.clear_hook.assert_called_with('pre-create')
        self.assertFalse(result)
        result = res.signal({'unset_hook': 'pre-update'})
        res.clear_hook.assert_called_with('pre-update')
        self.assertFalse(result)
        res.has_hook = mock.Mock(return_value=False)
        self.assertRaises(exception.InvalidBreakPointHook, res.signal, {'unset_hook': 'pre-create'})

    def test_pre_create_hook_call(self):
        self.stack.env.registry.load({'resources': {'res': {'hooks': 'pre-create'}}})
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.id = '1234'
        res.uuid = uuid.uuid4()
        res.store = mock.Mock()
        task = scheduler.TaskRunner(res.create)
        task.start()
        task.step()
        self.assertTrue(res.has_hook('pre-create'))
        res.signal(details={'unset_hook': 'pre-create'})
        task.run_to_completion()
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)

    def test_pre_delete_hook_call(self):
        self.stack.env.registry.load({'resources': {'res': {'hooks': 'pre-delete'}}})
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.id = '1234'
        res.action = 'CREATE'
        res.uuid = uuid.uuid4()
        res.store = mock.Mock()
        self.stack.action = 'DELETE'
        task = scheduler.TaskRunner(res.delete)
        task.start()
        task.step()
        self.assertTrue(res.has_hook('pre-delete'))
        res.signal(details={'unset_hook': 'pre-delete'})
        task.run_to_completion()
        self.assertEqual((res.DELETE, res.COMPLETE), res.state)

    def test_post_create_hook_call(self):
        self.stack.env.registry.load({'resources': {'res': {'hooks': 'post-create'}}})
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.id = '1234'
        res.uuid = uuid.uuid4()
        res.store = mock.Mock()
        task = scheduler.TaskRunner(res.create)
        task.start()
        task.step()
        self.assertTrue(res.has_hook('post-create'))
        res.signal(details={'unset_hook': 'post-create'})
        task.run_to_completion()
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)

    def test_post_delete_hook_call(self):
        self.stack.env.registry.load({'resources': {'res': {'hooks': 'post-delete'}}})
        snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
        res = resource.Resource('res', snippet, self.stack)
        res.id = '1234'
        res.uuid = uuid.uuid4()
        res.action = 'CREATE'
        self.stack.action = 'DELETE'
        res.store = mock.Mock()
        task = scheduler.TaskRunner(res.delete)
        task.start()
        task.step()
        self.assertTrue(res.has_hook('post-delete'))
        res.signal(details={'unset_hook': 'post-delete'})
        task.run_to_completion()
        self.assertEqual((res.DELETE, res.COMPLETE), res.state)