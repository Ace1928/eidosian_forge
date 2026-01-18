from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, InternetGateway
class TestDescribeInternetGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeInternetGatewaysResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <internetGatewaySet>\n                  <item>\n                     <internetGatewayId>igw-eaad4883EXAMPLE</internetGatewayId>\n                     <attachmentSet>\n                        <item>\n                           <vpcId>vpc-11ad4878</vpcId>\n                           <state>available</state>\n                        </item>\n                     </attachmentSet>\n                     <tagSet/>\n                  </item>\n               </internetGatewaySet>\n            </DescribeInternetGatewaysResponse>\n        '

    def test_describe_internet_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_internet_gateways('igw-eaad4883EXAMPLE', filters=[('attachment.state', ['available', 'pending'])])
        self.assert_request_parameters({'Action': 'DescribeInternetGateways', 'InternetGatewayId.1': 'igw-eaad4883EXAMPLE', 'Filter.1.Name': 'attachment.state', 'Filter.1.Value.1': 'available', 'Filter.1.Value.2': 'pending'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(len(api_response), 1)
        self.assertIsInstance(api_response[0], InternetGateway)
        self.assertEqual(api_response[0].id, 'igw-eaad4883EXAMPLE')