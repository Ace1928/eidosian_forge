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
def GetCredentialStore(store_file=None, access_token_file=None, with_access_token_cache=True):
    """Constructs credential store.

  Args:
    store_file: str, optional path to use for storage. If not specified
      config.Paths().credentials_path will be used.

    access_token_file: str, optional path to use for access token storage. Note
      that some implementations use store_file to also store access_tokens, in
      which case this argument is ignored.
    with_access_token_cache: bool, True to load a credential store with
      auto caching for access tokens. False to load a credential store without
      auto caching for access tokens.

  Returns:
    CredentialStore object.
  """
    if with_access_token_cache:
        return _GetSqliteStoreWithCache(store_file, access_token_file)
    else:
        return _GetSqliteStore(store_file)