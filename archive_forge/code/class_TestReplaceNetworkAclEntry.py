from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
class TestReplaceNetworkAclEntry(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <ReplaceNetworkAclEntryResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </ReplaceNetworkAclEntryResponse>\n        '

    def test_replace_network_acl(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.replace_network_acl_entry('acl-2cb85d45', 110, 'tcp', 'deny', '0.0.0.0/0', egress=False, port_range_from=139, port_range_to=139)
        self.assert_request_parameters({'Action': 'ReplaceNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'tcp', 'RuleAction': 'deny', 'Egress': 'false', 'CidrBlock': '0.0.0.0/0', 'PortRange.From': 139, 'PortRange.To': 139}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(response, True)

    def test_replace_network_acl_icmp(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.replace_network_acl_entry('acl-2cb85d45', 110, 'tcp', 'deny', '0.0.0.0/0', icmp_code=-1, icmp_type=8)
        self.assert_request_parameters({'Action': 'ReplaceNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'tcp', 'RuleAction': 'deny', 'CidrBlock': '0.0.0.0/0', 'Icmp.Code': -1, 'Icmp.Type': 8}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(response, True)