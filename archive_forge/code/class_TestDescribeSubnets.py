from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, Subnet
class TestDescribeSubnets(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeSubnetsResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <subnetSet>\n                <item>\n                  <subnetId>subnet-9d4a7b6c</subnetId>\n                  <state>available</state>\n                  <vpcId>vpc-1a2b3c4d</vpcId>\n                  <cidrBlock>10.0.1.0/24</cidrBlock>\n                  <availableIpAddressCount>251</availableIpAddressCount>\n                  <availabilityZone>us-east-1a</availabilityZone>\n                  <defaultForAz>false</defaultForAz>\n                  <mapPublicIpOnLaunch>false</mapPublicIpOnLaunch>\n                  <tagSet/>\n                </item>\n                <item>\n                  <subnetId>subnet-6e7f829e</subnetId>\n                  <state>available</state>\n                  <vpcId>vpc-1a2b3c4d</vpcId>\n                  <cidrBlock>10.0.0.0/24</cidrBlock>\n                  <availableIpAddressCount>251</availableIpAddressCount>\n                  <availabilityZone>us-east-1a</availabilityZone>\n                  <defaultForAz>false</defaultForAz>\n                  <mapPublicIpOnLaunch>false</mapPublicIpOnLaunch>\n                  <tagSet/>\n                </item>\n              </subnetSet>\n            </DescribeSubnetsResponse>\n        '

    def test_get_all_subnets(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_subnets(['subnet-9d4a7b6c', 'subnet-6e7f829e'], filters=OrderedDict([('state', 'available'), ('vpc-id', ['subnet-9d4a7b6c', 'subnet-6e7f829e'])]))
        self.assert_request_parameters({'Action': 'DescribeSubnets', 'SubnetId.1': 'subnet-9d4a7b6c', 'SubnetId.2': 'subnet-6e7f829e', 'Filter.1.Name': 'state', 'Filter.1.Value.1': 'available', 'Filter.2.Name': 'vpc-id', 'Filter.2.Value.1': 'subnet-9d4a7b6c', 'Filter.2.Value.2': 'subnet-6e7f829e'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(len(api_response), 2)
        self.assertIsInstance(api_response[0], Subnet)
        self.assertEqual(api_response[0].id, 'subnet-9d4a7b6c')
        self.assertEqual(api_response[1].id, 'subnet-6e7f829e')