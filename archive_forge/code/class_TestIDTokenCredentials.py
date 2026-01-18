import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
class TestIDTokenCredentials(object):
    SERVICE_ACCOUNT_EMAIL = 'service-account@example.com'
    TOKEN_URI = 'https://example.com/oauth2/token'
    TARGET_AUDIENCE = 'https://example.com'

    @classmethod
    def make_credentials(cls):
        return service_account.IDTokenCredentials(test_service_account.SIGNER, cls.SERVICE_ACCOUNT_EMAIL, cls.TOKEN_URI, cls.TARGET_AUDIENCE)

    def test_from_service_account_info(self):
        credentials = service_account.IDTokenCredentials.from_service_account_info(test_service_account.SERVICE_ACCOUNT_INFO, target_audience=self.TARGET_AUDIENCE)
        assert credentials._signer.key_id == test_service_account.SERVICE_ACCOUNT_INFO['private_key_id']
        assert credentials.service_account_email == test_service_account.SERVICE_ACCOUNT_INFO['client_email']
        assert credentials._token_uri == test_service_account.SERVICE_ACCOUNT_INFO['token_uri']
        assert credentials._target_audience == self.TARGET_AUDIENCE

    def test_from_service_account_file(self):
        info = test_service_account.SERVICE_ACCOUNT_INFO.copy()
        credentials = service_account.IDTokenCredentials.from_service_account_file(test_service_account.SERVICE_ACCOUNT_JSON_FILE, target_audience=self.TARGET_AUDIENCE)
        assert credentials.service_account_email == info['client_email']
        assert credentials._signer.key_id == info['private_key_id']
        assert credentials._token_uri == info['token_uri']
        assert credentials._target_audience == self.TARGET_AUDIENCE

    def test_default_state(self):
        credentials = self.make_credentials()
        assert not credentials.valid
        assert not credentials.expired

    def test_sign_bytes(self):
        credentials = self.make_credentials()
        to_sign = b'123'
        signature = credentials.sign_bytes(to_sign)
        assert crypt.verify_signature(to_sign, signature, test_service_account.PUBLIC_CERT_BYTES)

    def test_signer(self):
        credentials = self.make_credentials()
        assert isinstance(credentials.signer, crypt.Signer)

    def test_signer_email(self):
        credentials = self.make_credentials()
        assert credentials.signer_email == self.SERVICE_ACCOUNT_EMAIL

    def test_with_target_audience(self):
        credentials = self.make_credentials()
        new_credentials = credentials.with_target_audience('https://new.example.com')
        assert new_credentials._target_audience == 'https://new.example.com'

    def test_with_quota_project(self):
        credentials = self.make_credentials()
        new_credentials = credentials.with_quota_project('project-foo')
        assert new_credentials._quota_project_id == 'project-foo'

    def test__make_authorization_grant_assertion(self):
        credentials = self.make_credentials()
        token = credentials._make_authorization_grant_assertion()
        payload = jwt.decode(token, test_service_account.PUBLIC_CERT_BYTES)
        assert payload['iss'] == self.SERVICE_ACCOUNT_EMAIL
        assert payload['aud'] == service_account.service_account._GOOGLE_OAUTH2_TOKEN_ENDPOINT
        assert payload['target_audience'] == self.TARGET_AUDIENCE

    @mock.patch('google.oauth2._client_async.id_token_jwt_grant', autospec=True)
    @pytest.mark.asyncio
    async def test_refresh_success(self, id_token_jwt_grant):
        credentials = self.make_credentials()
        token = 'token'
        id_token_jwt_grant.return_value = (token, _helpers.utcnow() + datetime.timedelta(seconds=500), {})
        request = mock.AsyncMock(spec=['transport.Request'])
        await credentials.refresh(request)
        assert id_token_jwt_grant.called
        called_request, token_uri, assertion = id_token_jwt_grant.call_args[0]
        assert called_request == request
        assert token_uri == credentials._token_uri
        assert jwt.decode(assertion, test_service_account.PUBLIC_CERT_BYTES)
        assert credentials.token == token
        assert credentials.valid

    @mock.patch('google.oauth2._client_async.id_token_jwt_grant', autospec=True)
    @pytest.mark.asyncio
    async def test_before_request_refreshes(self, id_token_jwt_grant):
        credentials = self.make_credentials()
        token = 'token'
        id_token_jwt_grant.return_value = (token, _helpers.utcnow() + datetime.timedelta(seconds=500), None)
        request = mock.AsyncMock(spec=['transport.Request'])
        assert not credentials.valid
        await credentials.before_request(request, 'GET', 'http://example.com?a=1#3', {})
        assert id_token_jwt_grant.called
        assert credentials.valid