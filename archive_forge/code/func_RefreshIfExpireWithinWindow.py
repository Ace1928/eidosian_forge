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
def RefreshIfExpireWithinWindow(credentials, window):
    """Refreshes credentials if they will expire within a time window.

  Args:
    credentials: google.auth.credentials.Credentials or
      client.OAuth2Credentials, the credentials to refresh.
    window: string, The threshold of the remaining lifetime of the token which
      can trigger the refresh.
  """
    if c_creds.IsOauth2ClientCredentials(credentials):
        expiry = credentials.token_expiry
    else:
        expiry = credentials.expiry
    almost_expire = not expiry or _TokenExpiresWithinWindow(window, expiry)
    if almost_expire:
        Refresh(credentials)