from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
def GetFreshAccessToken(account=None, scopes=None, min_expiry_duration='1h', allow_account_impersonation=True):
    """Returns a fresh access token of the given account or the active account.

  Same as GetAccessToken except that the access token returned by
  this function is valid for at least min_expiry_duration.

  Args:
    account: str, The account to get the access token for. If None, the
      account stored in the core.account property is used.
    scopes: tuple, Custom auth scopes to request. By default CLOUDSDK_SCOPES are
      requested.
    min_expiry_duration: Duration str, Refresh the token if they are
      within this duration from expiration. Must be a valid duration between 0
      seconds and 1 hour (e.g. '0s' >x< '1h').
    allow_account_impersonation: bool, True to allow use of impersonated service
      account credentials (if that is configured).
  """
    creds = LoadFreshCredential(account, scopes, min_expiry_duration, allow_account_impersonation, True)
    return _GetAccessTokenFromCreds(creds)