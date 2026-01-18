from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGenerateCredentialReport(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n          <GenerateCredentialReportResponse>\n            <GenerateCredentialReportResult>\n              <State>COMPLETE</State>\n            </GenerateCredentialReportResult>\n            <ResponseMetadata>\n              <RequestId>b62e22a3-0da1-11e4-ba55-0990EXAMPLE</RequestId>\n            </ResponseMetadata>\n          </GenerateCredentialReportResponse>\n        '

    def test_generate_credential_report(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.generate_credential_report()
        self.assertEquals(response['generate_credential_report_response']['generate_credential_report_result']['state'], 'COMPLETE')