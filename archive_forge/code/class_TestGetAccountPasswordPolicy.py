from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetAccountPasswordPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <GetAccountPasswordPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n              <GetAccountPasswordPolicyResult>\n                <PasswordPolicy>\n                  <AllowUsersToChangePassword>true</AllowUsersToChangePassword>\n                  <RequireUppercaseCharacters>true</RequireUppercaseCharacters>\n                  <RequireSymbols>true</RequireSymbols>\n                  <ExpirePasswords>false</ExpirePasswords>\n                  <PasswordReusePrevention>12</PasswordReusePrevention>\n                  <RequireLowercaseCharacters>true</RequireLowercaseCharacters>\n                  <MaxPasswordAge>90</MaxPasswordAge>\n                  <HardExpiry>false</HardExpiry>\n                  <RequireNumbers>true</RequireNumbers>\n                  <MinimumPasswordLength>12</MinimumPasswordLength>\n                </PasswordPolicy>\n              </GetAccountPasswordPolicyResult>\n              <ResponseMetadata>\n                <RequestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</RequestId>\n              </ResponseMetadata>\n            </GetAccountPasswordPolicyResponse>\n        '

    def test_get_account_password_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_account_password_policy()
        self.assert_request_parameters({'Action': 'GetAccountPasswordPolicy'}, ignore_params_values=['Version'])
        self.assertEquals(response['get_account_password_policy_response']['get_account_password_policy_result']['password_policy']['minimum_password_length'], '12')