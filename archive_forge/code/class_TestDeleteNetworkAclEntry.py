from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
class TestDeleteNetworkAclEntry(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteNetworkAclEntryResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </DeleteNetworkAclEntryResponse>\n        '

    def test_delete_network_acl(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.delete_network_acl_entry('acl-2cb85d45', 100, egress=False)
        self.assert_request_parameters({'Action': 'DeleteNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 100, 'Egress': 'false'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(response, True)