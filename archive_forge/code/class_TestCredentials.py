import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
class TestCredentials:
    TOKEN_URI = 'https://example.com/oauth2/token'
    REFRESH_TOKEN = 'refresh_token'
    CLIENT_ID = 'client_id'
    CLIENT_SECRET = 'client_secret'

    @classmethod
    def make_credentials(cls):
        return _credentials_async.Credentials(token=None, refresh_token=cls.REFRESH_TOKEN, token_uri=cls.TOKEN_URI, client_id=cls.CLIENT_ID, client_secret=cls.CLIENT_SECRET, enable_reauth_refresh=True)

    def test_default_state(self):
        credentials = self.make_credentials()
        assert not credentials.valid
        assert not credentials.expired
        assert not credentials.requires_scopes
        assert credentials.refresh_token == self.REFRESH_TOKEN
        assert credentials.token_uri == self.TOKEN_URI
        assert credentials.client_id == self.CLIENT_ID
        assert credentials.client_secret == self.CLIENT_SECRET

    @mock.patch('google.oauth2._reauth_async.refresh_grant', autospec=True)
    @mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
    @pytest.mark.asyncio
    async def test_refresh_success(self, unused_utcnow, refresh_grant):
        token = 'token'
        expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
        grant_response = {'id_token': mock.sentinel.id_token}
        rapt_token = 'rapt_token'
        refresh_grant.return_value = (token, None, expiry, grant_response, rapt_token)
        request = mock.AsyncMock(spec=['transport.Request'])
        creds = self.make_credentials()
        await creds.refresh(request)
        refresh_grant.assert_called_with(request, self.TOKEN_URI, self.REFRESH_TOKEN, self.CLIENT_ID, self.CLIENT_SECRET, None, None, True)
        assert creds.token == token
        assert creds.expiry == expiry
        assert creds.id_token == mock.sentinel.id_token
        assert creds.rapt_token == rapt_token
        assert creds.valid

    @pytest.mark.asyncio
    async def test_refresh_no_refresh_token(self):
        request = mock.AsyncMock(spec=['transport.Request'])
        credentials_ = _credentials_async.Credentials(token=None, refresh_token=None)
        with pytest.raises(exceptions.RefreshError, match='necessary fields'):
            await credentials_.refresh(request)
        request.assert_not_called()

    @mock.patch('google.oauth2._reauth_async.refresh_grant', autospec=True)
    @mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
    @pytest.mark.asyncio
    async def test_credentials_with_scopes_requested_refresh_success(self, unused_utcnow, refresh_grant):
        scopes = ['email', 'profile']
        token = 'token'
        expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
        grant_response = {'id_token': mock.sentinel.id_token}
        rapt_token = 'rapt_token'
        refresh_grant.return_value = (token, None, expiry, grant_response, rapt_token)
        request = mock.AsyncMock(spec=['transport.Request'])
        creds = _credentials_async.Credentials(token=None, refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, scopes=scopes, rapt_token='old_rapt_token')
        await creds.refresh(request)
        refresh_grant.assert_called_with(request, self.TOKEN_URI, self.REFRESH_TOKEN, self.CLIENT_ID, self.CLIENT_SECRET, scopes, 'old_rapt_token', False)
        assert creds.token == token
        assert creds.expiry == expiry
        assert creds.id_token == mock.sentinel.id_token
        assert creds.has_scopes(scopes)
        assert creds.rapt_token == rapt_token
        assert creds.valid

    @mock.patch('google.oauth2._reauth_async.refresh_grant', autospec=True)
    @mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
    @pytest.mark.asyncio
    async def test_credentials_with_scopes_returned_refresh_success(self, unused_utcnow, refresh_grant):
        scopes = ['email', 'profile']
        token = 'token'
        expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
        grant_response = {'id_token': mock.sentinel.id_token, 'scope': ' '.join(scopes)}
        rapt_token = 'rapt_token'
        refresh_grant.return_value = (token, None, expiry, grant_response, rapt_token)
        request = mock.AsyncMock(spec=['transport.Request'])
        creds = _credentials_async.Credentials(token=None, refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, scopes=scopes)
        await creds.refresh(request)
        refresh_grant.assert_called_with(request, self.TOKEN_URI, self.REFRESH_TOKEN, self.CLIENT_ID, self.CLIENT_SECRET, scopes, None, False)
        assert creds.token == token
        assert creds.expiry == expiry
        assert creds.id_token == mock.sentinel.id_token
        assert creds.has_scopes(scopes)
        assert creds.rapt_token == rapt_token
        assert creds.valid

    @mock.patch('google.oauth2._reauth_async.refresh_grant', autospec=True)
    @mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
    @pytest.mark.asyncio
    async def test_credentials_with_scopes_refresh_failure_raises_refresh_error(self, unused_utcnow, refresh_grant):
        scopes = ['email', 'profile']
        scopes_returned = ['email']
        token = 'token'
        expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
        grant_response = {'id_token': mock.sentinel.id_token, 'scope': ' '.join(scopes_returned)}
        rapt_token = 'rapt_token'
        refresh_grant.return_value = (token, None, expiry, grant_response, rapt_token)
        request = mock.AsyncMock(spec=['transport.Request'])
        creds = _credentials_async.Credentials(token=None, refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, scopes=scopes, rapt_token=None)
        with pytest.raises(exceptions.RefreshError, match='Not all requested scopes were granted'):
            await creds.refresh(request)
        refresh_grant.assert_called_with(request, self.TOKEN_URI, self.REFRESH_TOKEN, self.CLIENT_ID, self.CLIENT_SECRET, scopes, None, False)
        assert creds.token == token
        assert creds.expiry == expiry
        assert creds.id_token == mock.sentinel.id_token
        assert creds.has_scopes(scopes)
        assert creds.valid

    def test_apply_with_quota_project_id(self):
        creds = _credentials_async.Credentials(token='token', refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, quota_project_id='quota-project-123')
        headers = {}
        creds.apply(headers)
        assert headers['x-goog-user-project'] == 'quota-project-123'

    def test_apply_with_no_quota_project_id(self):
        creds = _credentials_async.Credentials(token='token', refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET)
        headers = {}
        creds.apply(headers)
        assert 'x-goog-user-project' not in headers

    def test_with_quota_project(self):
        creds = _credentials_async.Credentials(token='token', refresh_token=self.REFRESH_TOKEN, token_uri=self.TOKEN_URI, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, quota_project_id='quota-project-123')
        new_creds = creds.with_quota_project('new-project-456')
        assert new_creds.quota_project_id == 'new-project-456'
        headers = {}
        creds.apply(headers)
        assert 'x-goog-user-project' in headers

    def test_from_authorized_user_info(self):
        info = test_credentials.AUTH_USER_INFO.copy()
        creds = _credentials_async.Credentials.from_authorized_user_info(info)
        assert creds.client_secret == info['client_secret']
        assert creds.client_id == info['client_id']
        assert creds.refresh_token == info['refresh_token']
        assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
        assert creds.scopes is None
        scopes = ['email', 'profile']
        creds = _credentials_async.Credentials.from_authorized_user_info(info, scopes)
        assert creds.client_secret == info['client_secret']
        assert creds.client_id == info['client_id']
        assert creds.refresh_token == info['refresh_token']
        assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
        assert creds.scopes == scopes

    def test_from_authorized_user_file(self):
        info = test_credentials.AUTH_USER_INFO.copy()
        creds = _credentials_async.Credentials.from_authorized_user_file(test_credentials.AUTH_USER_JSON_FILE)
        assert creds.client_secret == info['client_secret']
        assert creds.client_id == info['client_id']
        assert creds.refresh_token == info['refresh_token']
        assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
        assert creds.scopes is None
        scopes = ['email', 'profile']
        creds = _credentials_async.Credentials.from_authorized_user_file(test_credentials.AUTH_USER_JSON_FILE, scopes)
        assert creds.client_secret == info['client_secret']
        assert creds.client_id == info['client_id']
        assert creds.refresh_token == info['refresh_token']
        assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
        assert creds.scopes == scopes

    def test_to_json(self):
        info = test_credentials.AUTH_USER_INFO.copy()
        creds = _credentials_async.Credentials.from_authorized_user_info(info)
        json_output = creds.to_json()
        json_asdict = json.loads(json_output)
        assert json_asdict.get('token') == creds.token
        assert json_asdict.get('refresh_token') == creds.refresh_token
        assert json_asdict.get('token_uri') == creds.token_uri
        assert json_asdict.get('client_id') == creds.client_id
        assert json_asdict.get('scopes') == creds.scopes
        assert json_asdict.get('client_secret') == creds.client_secret
        json_output = creds.to_json(strip=['client_secret'])
        json_asdict = json.loads(json_output)
        assert json_asdict.get('token') == creds.token
        assert json_asdict.get('refresh_token') == creds.refresh_token
        assert json_asdict.get('token_uri') == creds.token_uri
        assert json_asdict.get('client_id') == creds.client_id
        assert json_asdict.get('scopes') == creds.scopes
        assert json_asdict.get('client_secret') is None

    def test_pickle_and_unpickle(self):
        creds = self.make_credentials()
        unpickled = pickle.loads(pickle.dumps(creds))
        assert list(creds.__dict__).sort() == list(unpickled.__dict__).sort()
        for attr in list(creds.__dict__):
            assert getattr(creds, attr) == getattr(unpickled, attr)

    def test_pickle_with_missing_attribute(self):
        creds = self.make_credentials()
        del creds.__dict__['_quota_project_id']
        unpickled = pickle.loads(pickle.dumps(creds))
        assert unpickled.quota_project_id is None

    @pytest.mark.skipif(sys.version_info < (3, 5), reason='pickle file can only be loaded with Python >= 3.5')
    def test_unpickle_old_credentials_pickle(self):
        with open(os.path.join(test_credentials.DATA_DIR, 'old_oauth_credentials_py3.pickle'), 'rb') as f:
            credentials = pickle.load(f)
            assert credentials.quota_project_id is None