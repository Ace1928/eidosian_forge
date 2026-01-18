import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
class GoogleOAuth2CredentialTest(GoogleTestCase):
    _ia_get_code_patcher = mock.patch('libcloud.common.google.GoogleInstalledAppAuthConnection.get_code', return_value=1234)

    def test_init_oauth2(self):
        kwargs = {'auth_type': GoogleAuthType.IA}
        cred = GoogleOAuth2Credential(*GCE_PARAMS, **kwargs)
        self.assertEqual(cred.token, STUB_TOKEN_FROM_FILE)
        with mock.patch.object(GoogleOAuth2Credential, '_get_token_from_file', return_value=None):
            cred = GoogleOAuth2Credential(*GCE_PARAMS, **kwargs)
            expected = STUB_IA_TOKEN
            expected['expire_time'] = cred.token['expire_time']
            self.assertEqual(cred.token, expected)
            cred._write_token_to_file.assert_called_once_with()

    def test_refresh(self):
        args = list(GCE_PARAMS) + [GoogleAuthType.GCE]
        cred = GoogleOAuth2Credential(*args)
        cred._refresh_token = mock.Mock()
        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
        cred.token = {'access_token': 'Access Token!', 'expire_time': _utc_timestamp(tomorrow)}
        cred.access_token
        self.assertFalse(cred._refresh_token.called)
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        cred.token = {'access_token': 'Access Token!', 'expire_time': _utc_timestamp(yesterday)}
        cred.access_token
        self.assertTrue(cred._refresh_token.called)

    def test_auth_connection(self):
        self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **{'auth_type': 'XX'})
        self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **{'auth_type': GoogleAuthType.GCS_S3})
        kwargs = {}
        if SHA256:
            kwargs['auth_type'] = GoogleAuthType.SA
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_PEM_KEY_FILE, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY_FILE, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_PEM_KEY, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_KEY, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            kwargs['auth_type'] = GoogleAuthType.SA
            cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY_STR, **kwargs)
            self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
            self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **kwargs)
            kwargs['auth_type'] = GoogleAuthType.SA
            expected_msg = 'Unable to decode provided PEM key:'
            self.assertRaisesRegex(GoogleAuthError, expected_msg, GoogleOAuth2Credential, *GCE_PARAMS_PEM_KEY_FILE_INVALID, **kwargs)
            kwargs['auth_type'] = GoogleAuthType.SA
            expected_msg = 'Unable to decode provided PEM key:'
            self.assertRaisesRegex(GoogleAuthError, expected_msg, GoogleOAuth2Credential, *GCE_PARAMS_JSON_KEY_INVALID, **kwargs)
        kwargs['auth_type'] = GoogleAuthType.IA
        cred2 = GoogleOAuth2Credential(*GCE_PARAMS_IA, **kwargs)
        self.assertTrue(isinstance(cred2.oauth2_conn, GoogleInstalledAppAuthConnection))
        kwargs['auth_type'] = GoogleAuthType.GCE
        cred3 = GoogleOAuth2Credential(*GCE_PARAMS_GCE, **kwargs)
        self.assertTrue(isinstance(cred3.oauth2_conn, GoogleGCEServiceAcctAuthConnection))