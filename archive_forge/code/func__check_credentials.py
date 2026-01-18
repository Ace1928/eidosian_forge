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
def _check_credentials(self, creds, store, access_token, refresh_token, token_expiry, invalid):
    stored_creds = store.locked_get() if store else creds
    self.assertEqual(access_token, creds.access_token)
    self.assertEqual(access_token, stored_creds.access_token)
    self.assertEqual(refresh_token, creds.refresh_token)
    self.assertEqual(refresh_token, stored_creds.refresh_token)
    self.assertEqual(token_expiry, creds.token_expiry)
    self.assertEqual(token_expiry, stored_creds.token_expiry)
    self.assertEqual(invalid, creds.invalid)
    self.assertEqual(invalid, stored_creds.invalid)