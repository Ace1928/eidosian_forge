from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, InternetGateway
class TestDetachInternetGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DetachInternetGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </DetachInternetGatewayResponse>\n        '

    def test_detach_internet_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.detach_internet_gateway('igw-eaad4883', 'vpc-11ad4878')
        self.assert_request_parameters({'Action': 'DetachInternetGateway', 'InternetGatewayId': 'igw-eaad4883', 'VpcId': 'vpc-11ad4878'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)