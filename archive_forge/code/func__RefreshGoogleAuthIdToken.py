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
def _RefreshGoogleAuthIdToken(credentials, is_impersonated_credential=False, include_email=False, gce_token_format='standard', gce_include_license=False, refresh_user_account_credentials=True):
    """Refreshes the ID token of google-auth credentials.

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
    refresh_user_account_credentials: bool, Specifies whether or not to refresh
      user account credentials. Note that when we refresh user account
      credentials access token, the ID token will be refreshed as well.
      Depending on where this function is called, we may not need to refresh
      user account credentials for ID token again.

  Raises:
    AccountImpersonationError: if impersonation support is not available for
      gcloud, or if the provided credentials is not google auth impersonation
      credentials.
  """
    import google.auth.compute_engine as google_auth_gce
    from google.oauth2 import service_account as google_auth_service_account
    from googlecloudsdk.core import requests
    request_client = requests.GoogleAuthRequest()
    with HandleGoogleAuthCredentialsRefreshError():
        id_token = None
        if c_creds.IsUserAccountCredentials(credentials) and refresh_user_account_credentials:
            credentials.refresh(request_client)
        elif is_impersonated_credential:
            if not IMPERSONATION_TOKEN_PROVIDER:
                raise AccountImpersonationError('gcloud is configured to impersonate a service account but impersonation support is not available.')
            import google.auth.impersonated_credentials as google_auth_impersonated_creds
            if not isinstance(credentials, google_auth_impersonated_creds.Credentials):
                raise AccountImpersonationError('Invalid impersonation account for refresh {}'.format(credentials))
            id_token_creds = IMPERSONATION_TOKEN_PROVIDER.GetElevationIdTokenGoogleAuth(credentials, config.CLOUDSDK_CLIENT_ID, include_email)
            id_token_creds.refresh(request_client)
            id_token = id_token_creds.token
        elif isinstance(credentials, google_auth_service_account.Credentials):
            id_token = _RefreshServiceAccountIdTokenGoogleAuth(credentials, request_client)
        elif isinstance(credentials, google_auth_gce.Credentials):
            id_token = c_gce.Metadata().GetIdToken(config.CLOUDSDK_CLIENT_ID, token_format=gce_token_format, include_license=gce_include_license)
        if id_token:
            credentials._id_token = id_token
            credentials.id_tokenb64 = id_token