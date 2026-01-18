import json
import os
from unittest import mock
from oslo_config import fixture as config_fixture
from heat.api.aws import exception
import heat.api.cfn.v1.stacks as stacks
from heat.common import exception as heat_exception
from heat.common import identifier
from heat.common import policy
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def _test_events_list(self, event_id):
    stack_name = 'wordpress'
    identity = dict(identifier.HeatIdentifier('t', stack_name, '6'))
    params = {'Action': 'DescribeStackEvents', 'StackName': stack_name}
    dummy_req = self._dummy_GET_request(params)
    self._stub_enforce(dummy_req, 'DescribeStackEvents')
    engine_resp = [{u'stack_name': u'wordpress', u'event_time': u'2012-07-23T13:05:39Z', u'stack_identity': {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'6', u'path': u''}, u'resource_name': u'WikiDatabase', u'resource_status_reason': u'state changed', u'event_identity': {u'tenant': u't', u'stack_name': u'wordpress', u'stack_id': u'6', u'path': u'/resources/WikiDatabase/events/{0}'.format(event_id)}, u'resource_action': u'TEST', u'resource_status': u'IN_PROGRESS', u'physical_resource_id': None, u'resource_properties': {u'UserData': u'blah'}, u'resource_type': u'AWS::EC2::Instance'}]
    kwargs = {'stack_identity': identity, 'nested_depth': None, 'limit': None, 'sort_keys': None, 'marker': None, 'sort_dir': None, 'filters': None}
    self.m_call.side_effect = [identity, engine_resp]
    response = self.controller.events_list(dummy_req)
    expected = {'DescribeStackEventsResponse': {'DescribeStackEventsResult': {'StackEvents': [{'EventId': str(event_id), 'StackId': u'arn:openstack:heat::t:stacks/wordpress/6', 'ResourceStatus': u'TEST_IN_PROGRESS', 'ResourceType': u'AWS::EC2::Instance', 'Timestamp': u'2012-07-23T13:05:39Z', 'StackName': u'wordpress', 'ResourceProperties': json.dumps({u'UserData': u'blah'}), 'PhysicalResourceId': None, 'ResourceStatusReason': u'state changed', 'LogicalResourceId': u'WikiDatabase'}]}}}
    self.assertEqual(expected, response)
    self.assertEqual([mock.call(dummy_req.context, ('identify_stack', {'stack_name': stack_name})), mock.call(dummy_req.context, ('list_events', kwargs), version='1.31')], self.m_call.call_args_list)