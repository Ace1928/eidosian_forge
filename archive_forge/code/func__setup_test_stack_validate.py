import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _setup_test_stack_validate(self, stack_name):
    t = template_format.parse(ipassoc_template_validate)
    template = tmpl.Template(t)
    stack = parser.Stack(utils.dummy_context(), stack_name, template, stack_id='12233', stack_user_project_id='8888')
    stack.validate()
    return (template, stack)