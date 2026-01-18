from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
class TestDescribeCustomerGateways(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeCustomerGatewaysResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <customerGatewaySet>\n                <item>\n                   <customerGatewayId>cgw-b4dc3961</customerGatewayId>\n                   <state>available</state>\n                   <type>ipsec.1</type>\n                   <ipAddress>12.1.2.3</ipAddress>\n                   <bgpAsn>65534</bgpAsn>\n                   <tagSet/>\n                </item>\n              </customerGatewaySet>\n            </DescribeCustomerGatewaysResponse>\n        '

    def test_get_all_customer_gateways(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_customer_gateways('cgw-b4dc3961', filters=OrderedDict([('state', ['pending', 'available']), ('ip-address', '12.1.2.3')]))
        self.assert_request_parameters({'Action': 'DescribeCustomerGateways', 'CustomerGatewayId.1': 'cgw-b4dc3961', 'Filter.1.Name': 'state', 'Filter.1.Value.1': 'pending', 'Filter.1.Value.2': 'available', 'Filter.2.Name': 'ip-address', 'Filter.2.Value.1': '12.1.2.3'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(len(api_response), 1)
        self.assertIsInstance(api_response[0], CustomerGateway)
        self.assertEqual(api_response[0].id, 'cgw-b4dc3961')