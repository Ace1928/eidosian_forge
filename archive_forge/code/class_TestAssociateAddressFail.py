from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
class TestAssociateAddressFail(TestEC2ConnectionBase):

    def default_body(self):
        return b"\n            <Response>\n                <Errors>\n                     <Error>\n                       <Code>InvalidInstanceID.NotFound</Code>\n                       <Message>The instance ID 'i-4cbc822a' does not exist</Message>\n                     </Error>\n                </Errors>\n                <RequestID>ea966190-f9aa-478e-9ede-cb5432daacc0</RequestID>\n                <StatusCode>Failure</StatusCode>\n            </Response>\n        "

    def test_associate_address(self):
        self.set_http_response(status_code=200)
        result = self.ec2.associate_address(instance_id='i-1234', public_ip='192.0.2.1')
        self.assertEqual(False, result)