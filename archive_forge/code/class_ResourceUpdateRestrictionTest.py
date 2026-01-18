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
class ResourceUpdateRestrictionTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceUpdateRestrictionTest, self).setUp()
        resource._register_class('TestResourceType', test_resource.TestResource)
        resource._register_class('NoneResourceType', none_resource.NoneResource)
        self.tmpl = {'heat_template_version': '2013-05-23', 'resources': {'bar': {'type': 'TestResourceType', 'properties': {'value': '1234', 'update_replace': False}}}}
        self.dummy_timeout = 10
        self.dummy_event = eventlet.event.Event()

    def create_resource(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(self.tmpl, env=self.env), stack_id=str(uuid.uuid4()))
        res = self.stack['bar']
        scheduler.TaskRunner(res.create)()
        return res

    def create_convergence_resource(self):
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(self.tmpl, env=self.env), stack_id=str(uuid.uuid4()))
        res = self.stack['bar']
        pcb = mock.Mock()
        self.patchobject(res, 'lock')
        res.create_convergence(self.stack.t.id, set(), 'engine-007', self.dummy_timeout, pcb)
        return res

    def test_update_restricted(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'update'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_resource()
        ev = self.patchobject(res, '_add_event')
        props = self.tmpl['resources']['bar']['properties']
        props['value'] = '4567'
        snippet = rsrc_defn.ResourceDefinition('bar', 'TestResourceType', props)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.update, snippet))
        self.assertEqual('ResourceActionRestricted: resources.bar: update is restricted for resource.', str(error))
        self.assertEqual('UPDATE', error.action)
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)
        ev.assert_called_with(res.UPDATE, res.FAILED, 'update is restricted for resource.')

    def test_replace_restricted(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'replace'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_resource()
        ev = self.patchobject(res, '_add_event')
        props = self.tmpl['resources']['bar']['properties']
        props['value'] = '4567'
        props['update_replace'] = True
        snippet = rsrc_defn.ResourceDefinition('bar', 'TestResourceType', props)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.update, snippet))
        self.assertEqual('ResourceActionRestricted: resources.bar: replace is restricted for resource.', str(error))
        self.assertEqual('UPDATE', error.action)
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)
        ev.assert_called_with(res.UPDATE, res.FAILED, 'replace is restricted for resource.')

    def test_update_with_replace_restricted(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'replace'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_resource()
        ev = self.patchobject(res, '_add_event')
        props = self.tmpl['resources']['bar']['properties']
        props['value'] = '4567'
        snippet = rsrc_defn.ResourceDefinition('bar', 'TestResourceType', props)
        self.assertIsNone(scheduler.TaskRunner(res.update, snippet)())
        self.assertEqual((res.UPDATE, res.COMPLETE), res.state)
        ev.assert_called_with(res.UPDATE, res.COMPLETE, 'state changed')

    def test_replace_with_update_restricted(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'update'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_resource()
        ev = self.patchobject(res, '_add_event')
        prep_replace = self.patchobject(res, 'prepare_for_replace')
        props = self.tmpl['resources']['bar']['properties']
        props['value'] = '4567'
        props['update_replace'] = True
        snippet = rsrc_defn.ResourceDefinition('bar', 'TestResourceType', props)
        error = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(res.update, snippet))
        self.assertIn('requires replacement', str(error))
        self.assertEqual(1, prep_replace.call_count)
        ev.assert_not_called()

    def test_replace_restricted_type_change_with_convergence(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'replace'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_convergence_resource()
        ev = self.patchobject(res, '_add_event')
        bar = self.tmpl['resources']['bar']
        bar['type'] = 'OS::Heat::None'
        self.new_stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(self.tmpl, env=self.env))
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.update_convergence, self.stack.t.id, set(), 'engine-007', self.dummy_timeout, self.new_stack, eventlet.event.Event()))
        self.assertEqual('ResourceActionRestricted: resources.bar: replace is restricted for resource.', str(error))
        self.assertEqual('UPDATE', error.action)
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)
        ev.assert_called_with(res.UPDATE, res.FAILED, 'replace is restricted for resource.')

    def test_update_restricted_type_change_with_convergence(self):
        self.env_snippet = {u'resource_registry': {u'resources': {'bar': {'restricted_actions': 'update'}}}}
        self.env = environment.Environment()
        self.env.load(self.env_snippet)
        res = self.create_convergence_resource()
        ev = self.patchobject(res, '_add_event')
        bar = self.tmpl['resources']['bar']
        bar['type'] = 'OS::Heat::None'
        self.new_stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(self.tmpl, env=self.env))
        error = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(res.update_convergence, self.stack.t.id, set(), 'engine-007', self.dummy_timeout, self.new_stack, eventlet.event.Event()))
        self.assertIn('requires replacement', str(error))
        ev.assert_not_called()