from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestAttachClassicLinkVpc(TestVpcClassicLink):

    def default_body(self):
        return b'\n            <AttachClassicLinkVpcResponse xmlns="http://ec2.amazonaws.com/doc/2014-09-01/">\n                <requestId>88673bdf-cd16-40bf-87a1-6132fec47257</requestId>\n                <return>true</return>\n            </AttachClassicLinkVpcResponse>\n        '

    def test_attach_classic_link_instance_string_groups(self):
        groups = ['sg-foo', 'sg-bar']
        self.set_http_response(status_code=200)
        response = self.vpc.attach_classic_instance(instance_id='my_instance_id', groups=groups, dry_run=True)
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'AttachClassicLinkVpc', 'VpcId': self.vpc_id, 'InstanceId': 'my_instance_id', 'SecurityGroupId.1': 'sg-foo', 'SecurityGroupId.2': 'sg-bar', 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])

    def test_attach_classic_link_instance_object_groups(self):
        sec_group_1 = SecurityGroup()
        sec_group_1.id = 'sg-foo'
        sec_group_2 = SecurityGroup()
        sec_group_2.id = 'sg-bar'
        groups = [sec_group_1, sec_group_2]
        self.set_http_response(status_code=200)
        response = self.vpc.attach_classic_instance(instance_id='my_instance_id', groups=groups, dry_run=True)
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'AttachClassicLinkVpc', 'VpcId': self.vpc_id, 'InstanceId': 'my_instance_id', 'SecurityGroupId.1': 'sg-foo', 'SecurityGroupId.2': 'sg-bar', 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])