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
class TestTerminateInstances(TestEC2ConnectionBase):

    def default_body(self):
        return b'<?xml version="1.0" ?>\n            <TerminateInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2013-07-15/">\n                <requestId>req-59a9ad52-0434-470c-ad48-4f89ded3a03e</requestId>\n                <instancesSet>\n                    <item>\n                        <instanceId>i-000043a2</instanceId>\n                        <shutdownState>\n                            <code>16</code>\n                            <name>running</name>\n                        </shutdownState>\n                        <previousState>\n                            <code>16</code>\n                            <name>running</name>\n                        </previousState>\n                    </item>\n                </instancesSet>\n            </TerminateInstancesResponse>\n        '

    def test_terminate_bad_response(self):
        self.set_http_response(status_code=200)
        self.ec2.terminate_instances('foo')