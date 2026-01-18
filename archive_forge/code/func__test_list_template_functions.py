import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def _test_list_template_functions(self, mock_enforce, req, engine_response, with_condition=False):
    self._mock_enforce_setup(mock_enforce, 'list_template_functions', True)
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_response)
    response = self.controller.list_template_functions(req, tenant_id=self.tenant, template_version='t1')
    self.assertEqual({'template_functions': engine_response}, response)
    mock_call.assert_called_once_with(req.context, ('list_template_functions', {'template_version': 't1', 'with_condition': with_condition}), version='1.35')