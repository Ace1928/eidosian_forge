from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestSetDefaultPolicyVersion(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<SetDefaultPolicyVersionResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <ResponseMetadata>\n    <RequestId>35f241af-3ebc-11e4-9d0d-6f969EXAMPLE</RequestId>\n  </ResponseMetadata>\n</SetDefaultPolicyVersionResponse>\n        '

    def test_set_default_policy_version(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.set_default_policy_version('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'v1')
        self.assert_request_parameters({'Action': 'SetDefaultPolicyVersion', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'VersionId': 'v1'}, ignore_params_values=['Version'])
        self.assertEqual('request_id' in response['set_default_policy_version_response']['response_metadata'], True)