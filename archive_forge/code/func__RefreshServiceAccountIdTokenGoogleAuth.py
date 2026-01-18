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
def _RefreshServiceAccountIdTokenGoogleAuth(cred, request_client):
    """Get a fresh id_token for the given google-auth credentials.

  Args:
    cred: google.oauth2.service_account.Credentials, the credentials for which
      to refresh the id_token.
    request_client: google.auth.transport.Request, the http transport
     to refresh with.

  Returns:
    str, The id_token if refresh was successful. Otherwise None.
  """
    if properties.VALUES.auth.service_account_disable_id_token_refresh.GetBool():
        return None
    from google.auth import exceptions as google_auth_exceptions
    from google.oauth2 import _client as google_auth_client
    from google.oauth2 import service_account as google_auth_service_account
    from googlecloudsdk.api_lib.iamcredentials import util as iam_credentials_util
    id_token_cred = google_auth_service_account.IDTokenCredentials(cred.signer, cred.service_account_email, cred._token_uri, config.CLOUDSDK_CLIENT_ID, universe_domain=properties.VALUES.core.universe_domain.Get())
    google_auth_client._IAM_IDTOKEN_ENDPOINT = google_auth_client._IAM_IDTOKEN_ENDPOINT.replace(iam_credentials_util.IAM_ENDPOINT_GDU, iam_credentials_util.GetEffectiveIamEndpoint())
    try:
        id_token_cred.refresh(request_client)
    except google_auth_exceptions.RefreshError:
        return None
    return id_token_cred.token