from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestReplaceRouteTableAssociation(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <ReplaceRouteTableAssociationResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <newAssociationId>rtbassoc-faad4893</newAssociationId>\n            </ReplaceRouteTableAssociationResponse>\n        '

    def test_replace_route_table_assocation(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.replace_route_table_assocation('rtbassoc-faad4893', 'rtb-f9ad4890')
        self.assert_request_parameters({'Action': 'ReplaceRouteTableAssociation', 'AssociationId': 'rtbassoc-faad4893', 'RouteTableId': 'rtb-f9ad4890'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)

    def test_replace_route_table_association_with_assoc(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.replace_route_table_association_with_assoc('rtbassoc-faad4893', 'rtb-f9ad4890')
        self.assert_request_parameters({'Action': 'ReplaceRouteTableAssociation', 'AssociationId': 'rtbassoc-faad4893', 'RouteTableId': 'rtb-f9ad4890'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, 'rtbassoc-faad4893')