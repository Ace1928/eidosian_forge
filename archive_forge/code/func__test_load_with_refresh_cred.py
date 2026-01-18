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
def _test_load_with_refresh_cred(self, refresh=True):
    cfg.CONF.set_override('deferred_auth_method', 'trusts')
    self.patchobject(self.ctx.auth_plugin, 'get_user_id', return_value='old_trustor_user_id')
    self.patchobject(self.ctx.auth_plugin, 'get_project_id', return_value='test_tenant_id')
    old_context = utils.dummy_context()
    old_context.trust_id = 'atrust123'
    old_context.trustor_user_id = 'trustor_user_id' if refresh else 'old_trustor_user_id'
    m_sc = self.patchobject(context, 'StoredContext')
    m_sc.from_dict.return_value = old_context
    self.stack = stack.Stack(self.ctx, 'test_regenerate_trust', self.tmpl)
    self.stack.store()
    load_stack = stack.Stack.load(self.ctx, stack_id=self.stack.id, check_refresh_cred=True)
    self.assertEqual(refresh, load_stack.refresh_cred)