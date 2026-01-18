from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.events as events
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def _test_show(self, event_id, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show', True)
    res_name = 'WikiDatabase'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    ev_identity = identifier.EventIdentifier(event_id=event_id, **res_identity)
    req = self._get(stack_identity._tenant_path() + '/resources/' + res_name + '/events/' + event_id)
    kwargs = {'stack_identity': stack_identity, 'limit': None, 'sort_keys': None, 'marker': None, 'sort_dir': None, 'nested_depth': None, 'filters': {'resource_name': res_name, 'uuid': event_id}}
    engine_resp = [{u'stack_name': u'wordpress', u'event_time': u'2012-07-23T13:06:00Z', u'stack_identity': dict(stack_identity), u'resource_name': res_name, u'resource_status_reason': u'state changed', u'event_identity': dict(ev_identity), u'resource_action': u'CREATE', u'resource_status': u'COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_properties': {u'UserData': u'blah'}, u'resource_type': u'AWS::EC2::Instance'}]
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_resp)
    result = self.controller.show(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name, event_id=event_id)
    expected = {'event': {'id': event_id, 'links': [{'href': self._url(ev_identity), 'rel': 'self'}, {'href': self._url(res_identity), 'rel': 'resource'}, {'href': self._url(stack_identity), 'rel': 'stack'}], u'resource_name': res_name, u'logical_resource_id': res_name, u'resource_status_reason': u'state changed', u'event_time': u'2012-07-23T13:06:00Z', u'resource_status': u'CREATE_COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_type': u'AWS::EC2::Instance', u'resource_properties': {u'UserData': u'blah'}}}
    self.assertEqual(expected, result)
    mock_call.assert_called_once_with(req.context, ('list_events', kwargs), version='1.31')