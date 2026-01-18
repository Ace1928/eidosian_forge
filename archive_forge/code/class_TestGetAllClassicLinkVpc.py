from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestGetAllClassicLinkVpc(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeVpcClassicLinkResponse xmlns="http://ec2.amazonaws.com/doc/2014-09-01/">\n                <requestId>2484655d-d669-4950-bf55-7ba559805d36</requestId>\n                <vpcSet>\n                    <item>\n                        <vpcId>vpc-6226ab07</vpcId>\n                        <classicLinkEnabled>false</classicLinkEnabled>\n                        <tagSet>\n                            <item>\n                                <key>Name</key>\n                                <value>hello</value>[\n                            </item>\n                        </tagSet>\n                    </item>\n                    <item>\n                        <vpcId>vpc-9d24f8f8</vpcId>\n                        <classicLinkEnabled>true</classicLinkEnabled>\n                        <tagSet/>\n                    </item>\n                </vpcSet>\n            </DescribeVpcClassicLinkResponse>\n        '

    def test_get_all_classic_link_vpcs(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_classic_link_vpcs()
        self.assertEqual(len(response), 2)
        vpc = response[0]
        self.assertEqual(vpc.id, 'vpc-6226ab07')
        self.assertEqual(vpc.classic_link_enabled, 'false')
        self.assertEqual(vpc.tags, {'Name': 'hello'})

    def test_get_all_classic_link_vpcs_params(self):
        self.set_http_response(status_code=200)
        self.service_connection.get_all_classic_link_vpcs(vpc_ids=['id1', 'id2'], filters={'GroupId': 'sg-9b4343fe'}, dry_run=True)
        self.assert_request_parameters({'Action': 'DescribeVpcClassicLink', 'VpcId.1': 'id1', 'VpcId.2': 'id2', 'Filter.1.Name': 'GroupId', 'Filter.1.Value.1': 'sg-9b4343fe', 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])