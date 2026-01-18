from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
class TestDetachVpnGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DetachVpnGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </DetachVpnGatewayResponse>\n        '

    def test_detach_vpn_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.detach_vpn_gateway('vgw-8db04f81', 'vpc-1a2b3c4d')
        self.assert_request_parameters({'Action': 'DetachVpnGateway', 'VpnGatewayId': 'vgw-8db04f81', 'VpcId': 'vpc-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(api_response, True)