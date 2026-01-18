import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationCancelUpdateStack(CloudFormationConnectionBase):

    def default_body(self):
        return b'<CancelUpdateStackResult/>'

    def test_cancel_update_stack(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.cancel_update_stack('stack_name')
        self.assertEqual(api_response, True)
        self.assert_request_parameters({'Action': 'CancelUpdateStack', 'StackName': 'stack_name', 'Version': '2010-05-15'})