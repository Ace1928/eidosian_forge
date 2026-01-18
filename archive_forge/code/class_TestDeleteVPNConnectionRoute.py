from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
class TestDeleteVPNConnectionRoute(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteVpnConnectionRouteResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n                <requestId>4f35a1b2-c2c3-4093-b51f-abb9d7311990</requestId>\n                <return>true</return>\n            </DeleteVpnConnectionRouteResponse>\n        '

    def test_delete_vpn_connection_route(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_vpn_connection_route('11.12.0.0/16', 'vpn-83ad48ea')
        self.assert_request_parameters({'Action': 'DeleteVpnConnectionRoute', 'DestinationCidrBlock': '11.12.0.0/16', 'VpnConnectionId': 'vpn-83ad48ea'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)