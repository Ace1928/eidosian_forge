from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestCreateVpc(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateVpcResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <vpc>\n                  <vpcId>vpc-1a2b3c4d</vpcId>\n                  <state>pending</state>\n                  <cidrBlock>10.0.0.0/16</cidrBlock>\n                  <dhcpOptionsId>dopt-1a2b3c4d2</dhcpOptionsId>\n                  <instanceTenancy>default</instanceTenancy>\n                  <tagSet/>\n               </vpc>\n            </CreateVpcResponse>\n        '

    def test_create_vpc(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_vpc('10.0.0.0/16', 'default')
        self.assert_request_parameters({'Action': 'CreateVpc', 'InstanceTenancy': 'default', 'CidrBlock': '10.0.0.0/16'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, VPC)
        self.assertEquals(api_response.id, 'vpc-1a2b3c4d')
        self.assertEquals(api_response.state, 'pending')
        self.assertEquals(api_response.cidr_block, '10.0.0.0/16')
        self.assertEquals(api_response.dhcp_options_id, 'dopt-1a2b3c4d2')
        self.assertEquals(api_response.instance_tenancy, 'default')