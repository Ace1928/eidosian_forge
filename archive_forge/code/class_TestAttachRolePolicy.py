from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestAttachRolePolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<AttachRolePolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>37a87673-3d07-11e4-bfad-8d1c6EXAMPLE</RequestId>\n  </ResponseMetadata>\n</AttachRolePolicyResponse>\n        '

    def test_attach_role_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.attach_role_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'DevRole')
        self.assert_request_parameters({'Action': 'AttachRolePolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'RoleName': 'DevRole'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['attach_role_policy_response']['response_metadata'], True)