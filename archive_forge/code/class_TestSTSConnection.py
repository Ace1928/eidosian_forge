from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
class TestSTSConnection(AWSMockServiceTestCase):
    connection_class = STSConnection

    def setUp(self):
        super(TestSTSConnection, self).setUp()

    def default_body(self):
        return b'\n            <AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">\n              <AssumeRoleResult>\n                <AssumedRoleUser>\n                  <Arn>arn:role</Arn>\n                  <AssumedRoleId>roleid:myrolesession</AssumedRoleId>\n                </AssumedRoleUser>\n                <Credentials>\n                  <SessionToken>session_token</SessionToken>\n                  <SecretAccessKey>secretkey</SecretAccessKey>\n                  <Expiration>2012-10-18T10:18:14.789Z</Expiration>\n                  <AccessKeyId>accesskey</AccessKeyId>\n                </Credentials>\n              </AssumeRoleResult>\n              <ResponseMetadata>\n                <RequestId>8b7418cb-18a8-11e2-a706-4bd22ca68ab7</RequestId>\n              </ResponseMetadata>\n            </AssumeRoleResponse>\n        '

    def test_assume_role(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.assume_role('arn:role', 'mysession')
        self.assert_request_parameters({'Action': 'AssumeRole', 'RoleArn': 'arn:role', 'RoleSessionName': 'mysession'}, ignore_params_values=['Version'])
        self.assertEqual(response.credentials.access_key, 'accesskey')
        self.assertEqual(response.credentials.secret_key, 'secretkey')
        self.assertEqual(response.credentials.session_token, 'session_token')
        self.assertEqual(response.user.arn, 'arn:role')
        self.assertEqual(response.user.assume_role_id, 'roleid:myrolesession')

    def test_assume_role_with_mfa(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.assume_role('arn:role', 'mysession', mfa_serial_number='GAHT12345678', mfa_token='abc123')
        self.assert_request_parameters({'Action': 'AssumeRole', 'RoleArn': 'arn:role', 'RoleSessionName': 'mysession', 'SerialNumber': 'GAHT12345678', 'TokenCode': 'abc123'}, ignore_params_values=['Version'])
        self.assertEqual(response.credentials.access_key, 'accesskey')
        self.assertEqual(response.credentials.secret_key, 'secretkey')
        self.assertEqual(response.credentials.session_token, 'session_token')
        self.assertEqual(response.user.arn, 'arn:role')
        self.assertEqual(response.user.assume_role_id, 'roleid:myrolesession')