from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
class TestEnableVgwRoutePropagation(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DisableVgwRoutePropagationResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n                <requestId>4f35a1b2-c2c3-4093-b51f-abb9d7311990</requestId>\n                <return>true</return>\n            </DisableVgwRoutePropagationResponse>\n        '

    def test_enable_vgw_route_propagation(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.enable_vgw_route_propagation('rtb-c98a35a0', 'vgw-d8e09e8a')
        self.assert_request_parameters({'Action': 'EnableVgwRoutePropagation', 'GatewayId': 'vgw-d8e09e8a', 'RouteTableId': 'rtb-c98a35a0'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(api_response, True)