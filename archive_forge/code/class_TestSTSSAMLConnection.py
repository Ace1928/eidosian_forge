from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
class TestSTSSAMLConnection(AWSMockServiceTestCase):
    connection_class = STSConnection

    def setUp(self):
        super(TestSTSSAMLConnection, self).setUp()

    def default_body(self):
        return b'\n<AssumeRoleWithSAMLResponse xmlns="https://sts.amazonaws.com/doc/\n2011-06-15/">\n  <AssumeRoleWithSAMLResult>\n    <Credentials>\n      <SessionToken>session_token</SessionToken>\n      <SecretAccessKey>secretkey</SecretAccessKey>\n      <Expiration>2011-07-15T23:28:33.359Z</Expiration>\n      <AccessKeyId>accesskey</AccessKeyId>\n    </Credentials>\n    <AssumedRoleUser>\n      <Arn>arn:role</Arn>\n      <AssumedRoleId>roleid:myrolesession</AssumedRoleId>\n    </AssumedRoleUser>\n    <PackedPolicySize>6</PackedPolicySize>\n  </AssumeRoleWithSAMLResult>\n  <ResponseMetadata>\n    <RequestId>c6104cbe-af31-11e0-8154-cbc7ccf896c7</RequestId>\n  </ResponseMetadata>\n</AssumeRoleWithSAMLResponse>\n'

    def test_assume_role_with_saml(self):
        arn = 'arn:aws:iam::000240903217:role/Test'
        principal = 'arn:aws:iam::000240903217:role/Principal'
        assertion = 'test'
        self.set_http_response(status_code=200)
        response = self.service_connection.assume_role_with_saml(role_arn=arn, principal_arn=principal, saml_assertion=assertion)
        self.assert_request_parameters({'RoleArn': arn, 'PrincipalArn': principal, 'SAMLAssertion': assertion, 'Action': 'AssumeRoleWithSAML'}, ignore_params_values=['Version'])
        self.assertEqual(response.credentials.access_key, 'accesskey')
        self.assertEqual(response.credentials.secret_key, 'secretkey')
        self.assertEqual(response.credentials.session_token, 'session_token')
        self.assertEqual(response.user.arn, 'arn:role')
        self.assertEqual(response.user.assume_role_id, 'roleid:myrolesession')