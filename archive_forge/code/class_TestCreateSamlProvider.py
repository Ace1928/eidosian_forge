from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestCreateSamlProvider(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <CreateSAMLProviderResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n              <CreateSAMLProviderResult>\n                <SAMLProviderArn>arn</SAMLProviderArn>\n              </CreateSAMLProviderResult>\n              <ResponseMetadata>\n                <RequestId>29f47818-99f5-11e1-a4c3-27EXAMPLE804</RequestId>\n              </ResponseMetadata>\n            </CreateSAMLProviderResponse>\n        '

    def test_create_saml_provider(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.create_saml_provider('document', 'name')
        self.assert_request_parameters({'Action': 'CreateSAMLProvider', 'SAMLMetadataDocument': 'document', 'Name': 'name'}, ignore_params_values=['Version'])
        self.assertEqual(response['create_saml_provider_response']['create_saml_provider_result']['saml_provider_arn'], 'arn')