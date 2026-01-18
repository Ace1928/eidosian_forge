from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestGetSamlProvider(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <GetSAMLProviderResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n              <GetSAMLProviderResult>\n                <CreateDate>2012-05-09T16:27:11Z</CreateDate>\n                <ValidUntil>2015-12-31T211:59:59Z</ValidUntil>\n                <SAMLMetadataDocument>Pd9fexDssTkRgGNqs...DxptfEs==</SAMLMetadataDocument>\n              </GetSAMLProviderResult>\n              <ResponseMetadata>\n                <RequestId>29f47818-99f5-11e1-a4c3-27EXAMPLE804</RequestId>\n              </ResponseMetadata>\n            </GetSAMLProviderResponse>\n        '

    def test_get_saml_provider(self):
        self.set_http_response(status_code=200)
        self.service_connection.get_saml_provider('arn')
        self.assert_request_parameters({'Action': 'GetSAMLProvider', 'SAMLProviderArn': 'arn'}, ignore_params_values=['Version'])