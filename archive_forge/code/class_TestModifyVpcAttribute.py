from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestModifyVpcAttribute(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <ModifyVpcAttributeResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </ModifyVpcAttributeResponse>\n        '

    def test_modify_vpc_attribute_dns_support(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.modify_vpc_attribute('vpc-1a2b3c4d', enable_dns_support=True)
        self.assert_request_parameters({'Action': 'ModifyVpcAttribute', 'VpcId': 'vpc-1a2b3c4d', 'EnableDnsSupport.Value': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)

    def test_modify_vpc_attribute_dns_hostnames(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.modify_vpc_attribute('vpc-1a2b3c4d', enable_dns_hostnames=True)
        self.assert_request_parameters({'Action': 'ModifyVpcAttribute', 'VpcId': 'vpc-1a2b3c4d', 'EnableDnsHostnames.Value': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)