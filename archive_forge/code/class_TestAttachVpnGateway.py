from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
class TestAttachVpnGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <AttachVpnGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <attachment>\n                  <vpcId>vpc-1a2b3c4d</vpcId>\n                  <state>attaching</state>\n               </attachment>\n            </AttachVpnGatewayResponse>\n        '

    def test_attach_vpn_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.attach_vpn_gateway('vgw-8db04f81', 'vpc-1a2b3c4d')
        self.assert_request_parameters({'Action': 'AttachVpnGateway', 'VpnGatewayId': 'vgw-8db04f81', 'VpcId': 'vpc-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, Attachment)
        self.assertEquals(api_response.vpc_id, 'vpc-1a2b3c4d')
        self.assertEquals(api_response.state, 'attaching')