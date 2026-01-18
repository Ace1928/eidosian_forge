from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestDeleteAccountPasswordPolicy(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <DeleteAccountPasswordPolicyResponse>\n              <ResponseMetadata>\n                <RequestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</RequestId>\n              </ResponseMetadata>\n            </DeleteAccountPasswordPolicyResponse>\n        '

    def test_delete_account_password_policy(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.delete_account_password_policy()
        self.assert_request_parameters({'Action': 'DeleteAccountPasswordPolicy'}, ignore_params_values=['Version'])