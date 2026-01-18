from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestDetachClassicLinkVpc(TestVpcClassicLink):

    def default_body(self):
        return b'\n            <DetachClassicLinkVpcResponse xmlns="http://ec2.amazonaws.com/doc/2014-09-01/">\n                <requestId>5565033d-1321-4eef-b121-6aa46f152ed7</requestId>\n                <return>true</return>\n            </DetachClassicLinkVpcResponse>\n        '

    def test_detach_classic_link_instance(self):
        self.set_http_response(status_code=200)
        response = self.vpc.detach_classic_instance(instance_id='my_instance_id', dry_run=True)
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'DetachClassicLinkVpc', 'VpcId': self.vpc_id, 'InstanceId': 'my_instance_id', 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])