from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestCreatePolicyVersion(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<CreatePolicyVersionResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <CreatePolicyVersionResult>\n    <PolicyVersion>\n      <IsDefaultVersion>true</IsDefaultVersion>\n      <VersionId>v2</VersionId>\n      <CreateDate>2014-09-15T19:58:59.430Z</CreateDate>\n    </PolicyVersion>\n  </CreatePolicyVersionResult>\n  <ResponseMetadata>\n    <RequestId>bb551b92-3d12-11e4-bfad-8d1c6EXAMPLE</RequestId>\n  </ResponseMetadata>\n</CreatePolicyVersionResponse>\n        '

    def test_create_policy_version(self):
        self.set_http_response(status_code=200)
        policy_doc = '\n{\n    "Version": "2012-10-17",\n    "Statement": [\n        {\n            "Sid": "Stmt1430948004000",\n            "Effect": "Deny",\n            "Action": [\n                "s3:*"\n            ],\n            "Resource": [\n                "*"\n            ]\n        }\n    ]\n}\n        '
        response = self.service_connection.create_policy_version('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', policy_doc, set_as_default=True)
        self.assert_request_parameters({'Action': 'CreatePolicyVersion', 'PolicyDocument': policy_doc, 'SetAsDefault': 'true', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
        self.assertEqual(response['create_policy_version_response']['create_policy_version_result']['policy_version']['is_default_version'], 'true')