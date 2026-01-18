from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_validation_failed(self, templatem, error_msg, error_cls=None):
    tmpl = template_format.parse(templatem)
    stack = utils.parse_stack(tmpl)
    rsrc_defns = stack.t.resource_definitions(stack)['workflow']
    wf = workflow.Workflow('workflow', rsrc_defns, stack)
    if error_cls is None:
        error_cls = exception.StackValidationFailed
    exc = self.assertRaises(error_cls, wf.validate)
    self.assertEqual(error_msg, str(exc))