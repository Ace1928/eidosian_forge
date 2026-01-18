from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetCredentialReport(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n          <GetCredentialReportResponse>\n            <ResponseMetadata>\n              <RequestId>99e60e9a-0db5-11e4-94d4-b764EXAMPLE</RequestId>\n            </ResponseMetadata>\n            <GetCredentialReportResult>\n              <Content>RXhhbXBsZQ==</Content>\n              <ReportFormat>text/csv</ReportFormat>\n              <GeneratedTime>2014-07-17T11:09:11Z</GeneratedTime>\n            </GetCredentialReportResult>\n          </GetCredentialReportResponse>\n        '

    def test_get_credential_report(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_credential_report()
        b64decode(response['get_credential_report_response']['get_credential_report_result']['content'])