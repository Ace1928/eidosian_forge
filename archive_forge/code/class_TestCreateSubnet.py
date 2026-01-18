from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, Subnet
class TestCreateSubnet(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateSubnetResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <subnet>\n                <subnetId>subnet-9d4a7b6c</subnetId>\n                <state>pending</state>\n                <vpcId>vpc-1a2b3c4d</vpcId>\n                <cidrBlock>10.0.1.0/24</cidrBlock>\n                <availableIpAddressCount>251</availableIpAddressCount>\n                <availabilityZone>us-east-1a</availabilityZone>\n                <tagSet/>\n              </subnet>\n            </CreateSubnetResponse>\n        '

    def test_create_subnet(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_subnet('vpc-1a2b3c4d', '10.0.1.0/24', 'us-east-1a')
        self.assert_request_parameters({'Action': 'CreateSubnet', 'VpcId': 'vpc-1a2b3c4d', 'CidrBlock': '10.0.1.0/24', 'AvailabilityZone': 'us-east-1a'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, Subnet)
        self.assertEquals(api_response.id, 'subnet-9d4a7b6c')
        self.assertEquals(api_response.state, 'pending')
        self.assertEquals(api_response.vpc_id, 'vpc-1a2b3c4d')
        self.assertEquals(api_response.cidr_block, '10.0.1.0/24')
        self.assertEquals(api_response.available_ip_address_count, 251)
        self.assertEquals(api_response.availability_zone, 'us-east-1a')