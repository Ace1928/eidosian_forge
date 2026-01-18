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
class ResourceDeleteRetryTest(common.HeatTestCase):

    def setUp(self):
        super(ResourceDeleteRetryTest, self).setUp()
        self.env = environment.Environment()
        self.env.load({u'resource_registry': {u'OS::Test::GenericResource': u'GenericResourceType'}})
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=self.env), stack_id=str(uuid.uuid4()))
        self.num_retries = 2
        cfg.CONF.set_override('action_retry_limit', self.num_retries)

    def test_delete_retry_conflict(self):
        tmpl = rsrc_defn.ResourceDefinition('test_resource', 'GenericResourceType', {'Foo': 'xyz123'})
        res = generic_rsrc.ResourceWithProps('test_resource', tmpl, self.stack)
        res.state_set(res.CREATE, res.COMPLETE, 'wobble')
        res.default_client_name = 'neutron'
        self.patchobject(timeutils, 'retry_backoff_delay', return_value=0.01)
        h_d_side_effects = [neutron_exp.Conflict(message='foo', request_ids=[1])] * (self.num_retries + 1)
        self.patchobject(generic_rsrc.GenericResource, 'handle_delete', side_effect=h_d_side_effects)
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.delete))
        exc_text = str(exc)
        self.assertIn('Conflict', exc_text)
        self.assertEqual(self.num_retries + 1, generic_rsrc.GenericResource.handle_delete.call_count)

    def test_delete_retry_phys_resource_exists(self):
        tmpl = rsrc_defn.ResourceDefinition('test_resource', 'Foo', {'Foo': 'abc'})
        res = generic_rsrc.ResourceWithPropsRefPropOnDelete('test_resource', tmpl, self.stack)
        res.state_set(res.CREATE, res.COMPLETE, 'wobble')
        cfg.CONF.set_override('action_retry_limit', self.num_retries)
        cdc_side_effects = [exception.PhysicalResourceExists(name='foo')] * self.num_retries
        cdc_side_effects.append(True)
        self.patchobject(timeutils, 'retry_backoff_delay', return_value=0.01)
        self.patchobject(generic_rsrc.GenericResource, 'handle_delete')
        self.patchobject(generic_rsrc.ResourceWithPropsRefPropOnDelete, 'check_delete_complete', side_effect=cdc_side_effects)
        scheduler.TaskRunner(res.delete)()
        self.assertEqual(self.num_retries + 1, generic_rsrc.GenericResource.handle_delete.call_count)