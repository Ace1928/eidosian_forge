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
class TestRegisterImage(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <RegisterImageResponse xmlns="http://ec2.amazonaws.com/doc/2013-08-15/">\n              <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n              <imageId>ami-1a2b3c4d</imageId>\n            </RegisterImageResponse>\n        '

    def test_vm_type_default(self):
        self.set_http_response(status_code=200)
        self.ec2.register_image('name', 'description', image_location='s3://foo')
        self.assert_request_parameters({'Action': 'RegisterImage', 'ImageLocation': 's3://foo', 'Name': 'name', 'Description': 'description'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_vm_type_hvm(self):
        self.set_http_response(status_code=200)
        self.ec2.register_image('name', 'description', image_location='s3://foo', virtualization_type='hvm')
        self.assert_request_parameters({'Action': 'RegisterImage', 'ImageLocation': 's3://foo', 'Name': 'name', 'Description': 'description', 'VirtualizationType': 'hvm'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_sriov_net_support_simple(self):
        self.set_http_response(status_code=200)
        self.ec2.register_image('name', 'description', image_location='s3://foo', sriov_net_support='simple')
        self.assert_request_parameters({'Action': 'RegisterImage', 'ImageLocation': 's3://foo', 'Name': 'name', 'Description': 'description', 'SriovNetSupport': 'simple'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_volume_delete_on_termination_on(self):
        self.set_http_response(status_code=200)
        self.ec2.register_image('name', 'description', snapshot_id='snap-12345678', delete_root_volume_on_termination=True)
        self.assert_request_parameters({'Action': 'RegisterImage', 'Name': 'name', 'Description': 'description', 'BlockDeviceMapping.1.DeviceName': None, 'BlockDeviceMapping.1.Ebs.DeleteOnTermination': 'true', 'BlockDeviceMapping.1.Ebs.SnapshotId': 'snap-12345678'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_volume_delete_on_termination_default(self):
        self.set_http_response(status_code=200)
        self.ec2.register_image('name', 'description', snapshot_id='snap-12345678')
        self.assert_request_parameters({'Action': 'RegisterImage', 'Name': 'name', 'Description': 'description', 'BlockDeviceMapping.1.DeviceName': None, 'BlockDeviceMapping.1.Ebs.DeleteOnTermination': 'false', 'BlockDeviceMapping.1.Ebs.SnapshotId': 'snap-12345678'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])