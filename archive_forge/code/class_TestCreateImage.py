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
class TestCreateImage(TestEC2ConnectionBase):

    def default_body(self):
        return b'<CreateImageResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n   <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n   <imageId>ami-4fa54026</imageId>\n</CreateImageResponse>'

    def test_minimal(self):
        self.set_http_response(status_code=200)
        response = self.ec2.create_image('instance_id', 'name')
        self.assert_request_parameters({'Action': 'CreateImage', 'InstanceId': 'instance_id', 'Name': 'name'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_block_device_mapping(self):
        self.set_http_response(status_code=200)
        bdm = BlockDeviceMapping()
        bdm['test'] = BlockDeviceType()
        response = self.ec2.create_image('instance_id', 'name', block_device_mapping=bdm)
        self.assert_request_parameters({'Action': 'CreateImage', 'InstanceId': 'instance_id', 'Name': 'name', 'BlockDeviceMapping.1.DeviceName': 'test', 'BlockDeviceMapping.1.Ebs.DeleteOnTermination': 'false'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])