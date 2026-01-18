from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestCreatePolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n<CreatePolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n  <CreatePolicyResult>\n    <Policy>\n      <PolicyName>S3-read-only-example-bucket</PolicyName>\n      <DefaultVersionId>v1</DefaultVersionId>\n      <PolicyId>AGPACKCEVSQ6C2EXAMPLE</PolicyId>\n      <Path>/</Path>\n      <Arn>arn:aws:iam::123456789012:policy/S3-read-only-example-bucket</Arn>\n      <AttachmentCount>0</AttachmentCount>\n      <CreateDate>2014-09-15T17:36:14.673Z</CreateDate>\n      <UpdateDate>2014-09-15T17:36:14.673Z</UpdateDate>\n    </Policy>\n  </CreatePolicyResult>\n  <ResponseMetadata>\n    <RequestId>ca64c9e1-3cfe-11e4-bfad-8d1c6EXAMPLE</RequestId>\n  </ResponseMetadata>\n</CreatePolicyResponse>\n        '

    def test_create_policy(self):
        self.set_http_response(status_code=200)
        policy_doc = '\n{\n    "Version": "2012-10-17",\n    "Statement": [\n        {\n            "Sid": "Stmt1430948004000",\n            "Effect": "Deny",\n            "Action": [\n                "s3:*"\n            ],\n            "Resource": [\n                "*"\n            ]\n        }\n    ]\n}\n        '
        response = self.service_connection.create_policy('S3-read-only-example-bucket', policy_doc)
        self.assert_request_parameters({'Action': 'CreatePolicy', 'PolicyDocument': policy_doc, 'Path': '/', 'PolicyName': 'S3-read-only-example-bucket'}, ignore_params_values=['Version'])
        self.assertEqual(response['create_policy_response']['create_policy_result']['policy']['policy_name'], 'S3-read-only-example-bucket')