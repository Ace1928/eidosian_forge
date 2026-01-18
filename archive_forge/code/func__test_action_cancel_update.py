import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.actions as actions
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def _test_action_cancel_update(self, mock_enforce, with_rollback=True):
    act = 'cancel_update' if with_rollback else 'cancel_without_rollback'
    self._mock_enforce_setup(mock_enforce, act, True)
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    body = {act: None}
    req = self._post(stack_identity._tenant_path() + '/actions', data=json.dumps(body))
    client = self.patchobject(rpc_client.EngineClient, 'call')
    result = self.controller.action(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, body=body)
    self.assertIsNone(result)
    client.assert_called_with(req.context, ('stack_cancel_update', {'stack_identity': stack_identity, 'cancel_with_rollback': with_rollback}), version='1.14')