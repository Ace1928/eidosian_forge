import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class StackStateSetTest(common.HeatTestCase):
    scenarios = [('in_progress', dict(action=stack.Stack.CREATE, status=stack.Stack.IN_PROGRESS, persist_count=1, error=False)), ('create_complete', dict(action=stack.Stack.CREATE, status=stack.Stack.COMPLETE, persist_count=0, error=False)), ('create_failed', dict(action=stack.Stack.CREATE, status=stack.Stack.FAILED, persist_count=0, error=False)), ('update_complete', dict(action=stack.Stack.UPDATE, status=stack.Stack.COMPLETE, persist_count=1, error=False)), ('update_failed', dict(action=stack.Stack.UPDATE, status=stack.Stack.FAILED, persist_count=1, error=False)), ('delete_complete', dict(action=stack.Stack.DELETE, status=stack.Stack.COMPLETE, persist_count=1, error=False)), ('delete_failed', dict(action=stack.Stack.DELETE, status=stack.Stack.FAILED, persist_count=1, error=False)), ('adopt_complete', dict(action=stack.Stack.ADOPT, status=stack.Stack.COMPLETE, persist_count=0, error=False)), ('adopt_failed', dict(action=stack.Stack.ADOPT, status=stack.Stack.FAILED, persist_count=0, error=False)), ('rollback_complete', dict(action=stack.Stack.ROLLBACK, status=stack.Stack.COMPLETE, persist_count=1, error=False)), ('rollback_failed', dict(action=stack.Stack.ROLLBACK, status=stack.Stack.FAILED, persist_count=1, error=False)), ('invalid_action', dict(action='action', status=stack.Stack.FAILED, persist_count=0, error=True)), ('invalid_status', dict(action=stack.Stack.CREATE, status='status', persist_count=0, error=True))]

    def test_state(self):
        self.tmpl = template.Template(copy.deepcopy(empty_template))
        self.ctx = utils.dummy_context()
        self.stack = stack.Stack(self.ctx, 'test_stack', self.tmpl, action=stack.Stack.CREATE, status=stack.Stack.IN_PROGRESS)
        persist_state = self.patchobject(self.stack, '_persist_state')
        self.assertEqual((stack.Stack.CREATE, stack.Stack.IN_PROGRESS), self.stack.state)
        if self.error:
            self.assertRaises(ValueError, self.stack.state_set, self.action, self.status, 'test')
        else:
            self.stack.state_set(self.action, self.status, 'test')
            self.assertEqual((self.action, self.status), self.stack.state)
            self.assertEqual('test', self.stack.status_reason)
        self.assertEqual(self.persist_count, persist_state.call_count)