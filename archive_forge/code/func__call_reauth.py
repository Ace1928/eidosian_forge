import base64
import json
import os
import unittest
import mock
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from google_reauth import challenges
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
from pyu2f import model
from pyu2f import u2f
def _call_reauth(self, request_mock, scopes=None):
    if os.environ.get('SK_SIGNING_PLUGIN') is not None:
        raise unittest.SkipTest('unset SK_SIGNING_PLUGIN.')
    return reauth.get_rapt_token(request_mock, self.client_id, self.client_secret, 'some_refresh_token', self.oauth_api_url, scopes=scopes)