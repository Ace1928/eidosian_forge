from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestDisassociateRouteTable(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DisassociateRouteTableResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </DisassociateRouteTableResponse>\n        '

    def test_disassociate_route_table(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.disassociate_route_table('rtbassoc-fdad4894')
        self.assert_request_parameters({'Action': 'DisassociateRouteTable', 'AssociationId': 'rtbassoc-fdad4894'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)