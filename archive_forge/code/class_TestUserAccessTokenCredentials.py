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
class TestUserAccessTokenCredentials(object):

    def test_instance(self):
        cred = _credentials_async.UserAccessTokenCredentials()
        assert cred._account is None
        cred = cred.with_account('account')
        assert cred._account == 'account'

    @mock.patch('google.auth._cloud_sdk.get_auth_access_token', autospec=True)
    def test_refresh(self, get_auth_access_token):
        get_auth_access_token.return_value = 'access_token'
        cred = _credentials_async.UserAccessTokenCredentials()
        cred.refresh(None)
        assert cred.token == 'access_token'

    def test_with_quota_project(self):
        cred = _credentials_async.UserAccessTokenCredentials()
        quota_project_cred = cred.with_quota_project('project-foo')
        assert quota_project_cred._quota_project_id == 'project-foo'
        assert quota_project_cred._account == cred._account

    @mock.patch('google.oauth2._credentials_async.UserAccessTokenCredentials.apply', autospec=True)
    @mock.patch('google.oauth2._credentials_async.UserAccessTokenCredentials.refresh', autospec=True)
    def test_before_request(self, refresh, apply):
        cred = _credentials_async.UserAccessTokenCredentials()
        cred.before_request(mock.Mock(), 'GET', 'https://example.com', {})
        refresh.assert_called()
        apply.assert_called()