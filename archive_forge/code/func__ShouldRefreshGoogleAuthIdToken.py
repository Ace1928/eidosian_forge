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
def _ShouldRefreshGoogleAuthIdToken(credentials):
    """Determine if ID token refresh is needed.

  (1) we don't refresh ID token for non-default universe domain.
  (2) for service account with self signed jwt feature enabled, we only refresh
  ID token if it's about to expire

  Args:
    credentials: google.auth.credentials.Credentials, A google-auth credentials
      to refresh.

  Returns:
    bool, Whether ID token refresh is needed.
  """
    from google.auth import exceptions as google_auth_exceptions
    from google.auth import jwt
    if not properties.IsDefaultUniverse() or not c_creds.HasDefaultUniverseDomain(credentials):
        return False
    if hasattr(credentials, '_id_token') and c_creds.UseSelfSignedJwt(credentials):
        try:
            payload = jwt.decode(credentials._id_token, verify=False)
        except google_auth_exceptions.GoogleAuthError:
            return True
        expiry = datetime.datetime.fromtimestamp(payload['exp'], tz=datetime.timezone.utc)
        if not _TokenExpiresWithinWindow(_CREDENTIALS_EXPIRY_WINDOW, expiry):
            return False
    return True