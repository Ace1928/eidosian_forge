from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, Subnet
class TestDeleteSubnet(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteSubnetResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </DeleteSubnetResponse>\n        '

    def test_delete_subnet(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_subnet('subnet-9d4a7b6c')
        self.assert_request_parameters({'Action': 'DeleteSubnet', 'SubnetId': 'subnet-9d4a7b6c'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)