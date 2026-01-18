from unittest import mock
from oslo_config import cfg
import uuid
from heat.db import api as db_api
from heat.db import models
from heat.engine import event
from heat.engine import stack
from heat.engine import template
from heat.objects import event as event_object
from heat.objects import resource_properties_data as rpd_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def _setup_stack(self, the_tmpl, encrypted=False):
    if encrypted:
        cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    self.username = 'event_test_user'
    self.ctx = utils.dummy_context()
    self.stack = stack.Stack(self.ctx, 'event_load_test_stack', template.Template(the_tmpl))
    self.stack.store()
    self.resource = self.stack['EventTestResource']
    self.resource._update_stored_properties()
    self.resource.store()
    self.addCleanup(stack_object.Stack.delete, self.ctx, self.stack.id)