from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestDetachRolePolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<DetachRolePolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>4c80ccf4-3d1e-11e4-a4a0-cffb9EXAMPLE</RequestId>\n  </ResponseMetadata>\n</DetachRolePolicyResponse>\n        '

    def test_detach_role_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.detach_role_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'DevRole')
        self.assert_request_parameters({'Action': 'DetachRolePolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'RoleName': 'DevRole'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['detach_role_policy_response']['response_metadata'], True)