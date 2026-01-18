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
def RevokeCredentials(credentials):
    """Revokes the token on the server.

  Args:
    credentials: user account credentials from either google-auth or
      oauth2client.
  Raises:
    RevokeError: If credentials to revoke is not user account credentials.
  """
    if not c_creds.IsUserAccountCredentials(credentials) or c_creds.IsExternalAccountUserCredentials(credentials) or c_creds.IsExternalAccountAuthorizedUserCredentials(credentials):
        raise RevokeError('The token cannot be revoked from server because it is not user account credentials.')
    if c_creds.IsOauth2ClientCredentials(credentials):
        from googlecloudsdk.core import http
        credentials.revoke(http.Http())
    else:
        from googlecloudsdk.core import requests
        credentials.revoke(requests.GoogleAuthRequest())