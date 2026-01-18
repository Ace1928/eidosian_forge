import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def _get_formatted_resource_properties(self, res_name):
    tmpl = template.Template(template_format.parse('\n            heat_template_version: 2013-05-23\n            resources:\n              resource1:\n                type: ResWithComplexPropsAndAttrs\n              resource2:\n                type: ResWithComplexPropsAndAttrs\n                properties:\n                  a_string: foobar\n              resource3:\n                type: ResWithComplexPropsAndAttrs\n                properties:\n                  a_string: { get_attr: [ resource2, string] }\n        '))
    stack = parser.Stack(utils.dummy_context(), 'test_stack_for_preview', tmpl, stack_id=str(uuid.uuid4()))
    res = stack[res_name]
    return api.format_resource_properties(res)