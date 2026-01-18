from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestAttachUserPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<AttachUserPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>ed7e72d3-3d07-11e4-bfad-8d1c6EXAMPLE</RequestId>\n  </ResponseMetadata>\n</AttachUserPolicyResponse>\n        '

    def test_attach_user_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.attach_user_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'Alice')
        self.assert_request_parameters({'Action': 'AttachUserPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'UserName': 'Alice'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['attach_user_policy_response']['response_metadata'], True)