from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def _setup_test_instance(self, name, t, rsrc_name='MySqlCloudDB'):
    stack_name = '%s_stack' % name
    template = tmpl.Template(t)
    self.stack = parser.Stack(utils.dummy_context(), stack_name, template, stack_id=str(uuid.uuid4()))
    rsrc = self.stack[rsrc_name]
    rsrc.resource_id = '12345'
    return rsrc