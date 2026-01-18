from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestDeleteRouteTable(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteRouteTableResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </DeleteRouteTableResponse>\n        '

    def test_delete_route_table(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_route_table('rtb-e4ad488d')
        self.assert_request_parameters({'Action': 'DeleteRouteTable', 'RouteTableId': 'rtb-e4ad488d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)