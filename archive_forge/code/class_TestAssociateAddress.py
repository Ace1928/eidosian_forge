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
class TestAssociateAddress(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <AssociateAddressResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-15/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n               <associationId>eipassoc-fc5ca095</associationId>\n            </AssociateAddressResponse>\n        '

    def test_associate_address(self):
        self.set_http_response(status_code=200)
        result = self.ec2.associate_address(instance_id='i-1234', public_ip='192.0.2.1')
        self.assertEqual(True, result)

    def test_associate_address_object(self):
        self.set_http_response(status_code=200)
        result = self.ec2.associate_address_object(instance_id='i-1234', public_ip='192.0.2.1')
        self.assertEqual('eipassoc-fc5ca095', result.association_id)