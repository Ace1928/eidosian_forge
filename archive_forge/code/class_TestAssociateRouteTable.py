from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestAssociateRouteTable(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <AssociateRouteTableResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <associationId>rtbassoc-f8ad4891</associationId>\n            </AssociateRouteTableResponse>\n        '

    def test_associate_route_table(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.associate_route_table('rtb-e4ad488d', 'subnet-15ad487c')
        self.assert_request_parameters({'Action': 'AssociateRouteTable', 'RouteTableId': 'rtb-e4ad488d', 'SubnetId': 'subnet-15ad487c'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, 'rtbassoc-f8ad4891')