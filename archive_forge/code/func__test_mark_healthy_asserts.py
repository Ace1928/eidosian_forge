from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def _test_mark_healthy_asserts(self, action='CHECK', status='FAILED', reason='state changed', meta=None):
    rs = self.eng.describe_stack_resource(self.ctx, self.stack.identifier(), 'WebServer', with_attr=None)
    self.assertIn('resource_action', rs)
    self.assertIn('resource_status', rs)
    self.assertIn('resource_status_reason', rs)
    self.assertEqual(action, rs['resource_action'])
    self.assertEqual(status, rs['resource_status'])
    self.assertEqual(reason, rs['resource_status_reason'])
    if meta is not None:
        self.assertIn('metadata', rs)
        self.assertEqual(meta, rs['metadata'])