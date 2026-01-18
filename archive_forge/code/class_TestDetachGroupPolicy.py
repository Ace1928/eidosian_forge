from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestDetachGroupPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<DetachGroupPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>d4faa7aa-3d1d-11e4-a4a0-cffb9EXAMPLE</RequestId>\n  </ResponseMetadata>\n</DetachGroupPolicyResponse>\n        '

    def test_detach_group_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.detach_group_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'Dev')
        self.assert_request_parameters({'Action': 'DetachGroupPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'GroupName': 'Dev'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['detach_group_policy_response']['response_metadata'], True)