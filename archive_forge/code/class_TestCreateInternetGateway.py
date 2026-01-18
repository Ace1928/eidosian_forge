from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, InternetGateway
class TestCreateInternetGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateInternetGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <internetGateway>\n                  <internetGatewayId>igw-eaad4883</internetGatewayId>\n                  <attachmentSet/>\n                  <tagSet/>\n               </internetGateway>\n            </CreateInternetGatewayResponse>\n        '

    def test_create_internet_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_internet_gateway()
        self.assert_request_parameters({'Action': 'CreateInternetGateway'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, InternetGateway)
        self.assertEqual(api_response.id, 'igw-eaad4883')