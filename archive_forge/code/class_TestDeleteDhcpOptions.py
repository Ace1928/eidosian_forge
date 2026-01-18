from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, DhcpOptions
class TestDeleteDhcpOptions(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteDhcpOptionsResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </DeleteDhcpOptionsResponse>\n        '

    def test_delete_dhcp_options(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_dhcp_options('dopt-7a8b9c2d')
        self.assert_request_parameters({'Action': 'DeleteDhcpOptions', 'DhcpOptionsId': 'dopt-7a8b9c2d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)