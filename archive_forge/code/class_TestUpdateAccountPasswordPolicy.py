from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestUpdateAccountPasswordPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <UpdateAccountPasswordPolicyResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n               <ResponseMetadata>\n                  <RequestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</RequestId>\n               </ResponseMetadata>\n            </UpdateAccountPasswordPolicyResponse>\n        '

    def test_update_account_password_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.update_account_password_policy(minimum_password_length=88)
        self.assert_request_parameters({'Action': 'UpdateAccountPasswordPolicy', 'MinimumPasswordLength': 88}, ignore_params_values=['Version'])