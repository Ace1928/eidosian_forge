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
class TestDescribeTags(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <DescribeTagsResponse>\n            </DescribeTagsResponse>\n        '

    def test_default_behavior(self):
        self.set_http_response(status_code=200)
        self.ec2.get_all_tags()
        self.assert_request_parameters({'Action': 'DescribeTags'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_max_results(self):
        self.set_http_response(status_code=200)
        self.ec2.get_all_tags(max_results=10)
        self.assert_request_parameters({'Action': 'DescribeTags', 'MaxResults': 10}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])