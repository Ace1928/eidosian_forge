import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationGetStackevents(CloudFormationConnectionBase):

    def default_body(self):
        return b'\n            <DescribeStackEventsResult>\n              <StackEvents>\n                <member>\n                  <EventId>Event-1-Id</EventId>\n                  <StackId>arn:aws:cfn:us-east-1:1:stack</StackId>\n                  <StackName>MyStack</StackName>\n                  <LogicalResourceId>MyStack</LogicalResourceId>\n                  <PhysicalResourceId>MyStack_One</PhysicalResourceId>\n                  <ResourceType>AWS::CloudFormation::Stack</ResourceType>\n                  <Timestamp>2010-07-27T22:26:28Z</Timestamp>\n                  <ResourceStatus>CREATE_IN_PROGRESS</ResourceStatus>\n                  <ResourceStatusReason>User initiated</ResourceStatusReason>\n                </member>\n                <member>\n                  <EventId>Event-2-Id</EventId>\n                  <StackId>arn:aws:cfn:us-east-1:1:stack</StackId>\n                  <StackName>MyStack</StackName>\n                  <LogicalResourceId>MySG1</LogicalResourceId>\n                  <PhysicalResourceId>MyStack_SG1</PhysicalResourceId>\n                  <ResourceType>AWS::SecurityGroup</ResourceType>\n                  <Timestamp>2010-07-27T22:28:28Z</Timestamp>\n                  <ResourceStatus>CREATE_COMPLETE</ResourceStatus>\n                </member>\n              </StackEvents>\n            </DescribeStackEventsResult>\n        '

    def test_describe_stack_events(self):
        self.set_http_response(status_code=200)
        first, second = self.service_connection.describe_stack_events('stack_name', next_token='next_token')
        self.assertEqual(first.event_id, 'Event-1-Id')
        self.assertEqual(first.logical_resource_id, 'MyStack')
        self.assertEqual(first.physical_resource_id, 'MyStack_One')
        self.assertEqual(first.resource_properties, None)
        self.assertEqual(first.resource_status, 'CREATE_IN_PROGRESS')
        self.assertEqual(first.resource_status_reason, 'User initiated')
        self.assertEqual(first.resource_type, 'AWS::CloudFormation::Stack')
        self.assertEqual(first.stack_id, 'arn:aws:cfn:us-east-1:1:stack')
        self.assertEqual(first.stack_name, 'MyStack')
        self.assertIsNotNone(first.timestamp)
        self.assertEqual(second.event_id, 'Event-2-Id')
        self.assertEqual(second.logical_resource_id, 'MySG1')
        self.assertEqual(second.physical_resource_id, 'MyStack_SG1')
        self.assertEqual(second.resource_properties, None)
        self.assertEqual(second.resource_status, 'CREATE_COMPLETE')
        self.assertEqual(second.resource_status_reason, None)
        self.assertEqual(second.resource_type, 'AWS::SecurityGroup')
        self.assertEqual(second.stack_id, 'arn:aws:cfn:us-east-1:1:stack')
        self.assertEqual(second.stack_name, 'MyStack')
        self.assertIsNotNone(second.timestamp)
        self.assert_request_parameters({'Action': 'DescribeStackEvents', 'NextToken': 'next_token', 'StackName': 'stack_name', 'Version': '2010-05-15'})