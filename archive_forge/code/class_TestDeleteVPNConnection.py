from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
class TestDeleteVPNConnection(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteVpnConnectionResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </DeleteVpnConnectionResponse>\n        '

    def test_delete_vpn_connection(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_vpn_connection('vpn-44a8938f')
        self.assert_request_parameters({'Action': 'DeleteVpnConnection', 'VpnConnectionId': 'vpn-44a8938f'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)