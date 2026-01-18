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
class TestCopyImage(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n        <CopyImageResponse xmlns="http://ec2.amazonaws.com/doc/2013-07-15/">\n           <requestId>request_id</requestId>\n           <imageId>ami-copied-id</imageId>\n        </CopyImageResponse>\n        '

    def test_copy_image_required_params(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id')
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_copy_image_name_and_description(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id', 'name', 'description')
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id', 'Name': 'name', 'Description': 'description'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_copy_image_client_token(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id', client_token='client-token')
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id', 'ClientToken': 'client-token'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_copy_image_encrypted(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id', encrypted=True)
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id', 'Encrypted': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_copy_image_not_encrypted(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id', encrypted=False)
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id', 'Encrypted': 'false'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_copy_image_encrypted_with_kms_key(self):
        self.set_http_response(status_code=200)
        copied_ami = self.ec2.copy_image('us-west-2', 'ami-id', encrypted=False, kms_key_id='kms-key')
        self.assertEqual(copied_ami.image_id, 'ami-copied-id')
        self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id', 'Encrypted': 'false', 'KmsKeyId': 'kms-key'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])