import base64
import datetime
import json
import os
import unittest
import mock
from mock import patch
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from oauth2client import client
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
def _run_refresh_test(self, http_mock, access_token, refresh_token, token_expiry, invalid):
    creds = self._get_creds()
    store = MockStore()
    creds.set_store(store)
    creds._do_refresh_request(http_mock)
    self._check_credentials(creds, store, access_token, refresh_token, token_expiry, invalid)