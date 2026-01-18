from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestDetachUserPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<DetachUserPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>85ba31fa-3d1f-11e4-a4a0-cffb9EXAMPLE</RequestId>\n  </ResponseMetadata>\n</DetachUserPolicyResponse>\n        '

    def test_detach_user_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.detach_user_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'Alice')
        self.assert_request_parameters({'Action': 'DetachUserPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'UserName': 'Alice'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['detach_user_policy_response']['response_metadata'], True)