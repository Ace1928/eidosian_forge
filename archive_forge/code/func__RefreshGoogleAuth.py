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
def _RefreshGoogleAuth(credentials, is_impersonated_credential=False, include_email=False, gce_token_format='standard', gce_include_license=False):
    """Refreshes google-auth credentials.

  Args:
    credentials: google.auth.credentials.Credentials, A google-auth credentials
      to refresh.
    is_impersonated_credential: bool, True treat provided credential as an
      impersonated service account credential. If False, treat as service
      account or user credential. Needed to avoid circular dependency on
      IMPERSONATION_TOKEN_PROVIDER.
    include_email: bool, Specifies whether or not the service account email is
      included in the identity token. Only applicable to impersonated service
      account.
    gce_token_format: str, Specifies whether or not the project and instance
      details are included in the identity token. Choices are "standard",
      "full".
    gce_include_license: bool, Specifies whether or not license codes for images
      associated with GCE instance are included in their identity tokens.

  Raises:
    AccountImpersonationError: if impersonation support is not available for
      gcloud, or if the provided credentials is not google auth impersonation
      credentials.
  """
    from googlecloudsdk.core import requests
    request_client = requests.GoogleAuthRequest()
    with HandleGoogleAuthCredentialsRefreshError():
        c_creds.EnableSelfSignedJwtIfApplicable(credentials)
        if _ShouldRefreshGoogleAuthIdToken(credentials):
            _RefreshGoogleAuthIdToken(credentials, is_impersonated_credential=is_impersonated_credential, include_email=include_email, gce_token_format=gce_token_format, gce_include_license=gce_include_license, refresh_user_account_credentials=False)
        credentials.refresh(request_client)