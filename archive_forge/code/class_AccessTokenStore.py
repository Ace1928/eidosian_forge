from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
class AccessTokenStore(client.Storage):
    """Oauth2client adapted for access token cache.

  This class works with Oauth2client model where access token is part of
  credential serialization format and get captured as part of that.
  By extending client.Storage this class pretends to serialize credentials, but
  only serializes access token.

  When fetching the more recent credentials from the cache, this does not return
  token_response, as it is now out of date.
  """

    def __init__(self, access_token_cache, account_id, credentials):
        """Sets up token store for given acount.

    Args:
      access_token_cache: AccessTokenCache, cache for access tokens.
      account_id: str, account for which token is stored.
      credentials: oauth2client.client.OAuth2Credentials, they are auto-updated
        with cached access token.
    """
        super(AccessTokenStore, self).__init__(lock=None)
        self._access_token_cache = access_token_cache
        self._account_id = account_id
        self._credentials = credentials

    def locked_get(self):
        token_data = self._access_token_cache.Load(self._account_id)
        if token_data:
            access_token, token_expiry, rapt_token, id_token = token_data
            self._credentials.access_token = access_token
            self._credentials.token_expiry = token_expiry
            if rapt_token is not None:
                self._credentials.rapt_token = rapt_token
            self._credentials.id_tokenb64 = id_token
            self._credentials.token_response = None
        return self._credentials

    def locked_put(self, credentials):
        if getattr(self._credentials, 'token_response'):
            id_token = self._credentials.token_response.get('id_token', None)
        else:
            id_token = None
        self._access_token_cache.Store(self._account_id, self._credentials.access_token, self._credentials.token_expiry, getattr(self._credentials, 'rapt_token', None), id_token)

    def locked_delete(self):
        self._access_token_cache.Remove(self._account_id)