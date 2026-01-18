from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import http
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
from oauth2client import client as oauth2client_client
from oauth2client.contrib import reauth
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials as google_auth_credentials
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import _client as google_auth_client
from google.oauth2 import credentials
from google.oauth2 import reauth as google_auth_reauth
def _RefreshGrant(request, token_uri, refresh_token, client_id, client_secret, scopes=None, rapt_token=None):
    """Prepares the request to send to auth server to refresh tokens."""
    body = [('grant_type', google_auth_client._REFRESH_GRANT_TYPE), ('client_id', client_id), ('client_secret', client_secret), ('refresh_token', refresh_token)]
    if scopes:
        body.append(('scope', ' '.join(scopes)))
    if rapt_token:
        body.append(('rapt', rapt_token))
    response_data = _TokenEndpointRequestWithRetry(request, token_uri, body)
    try:
        access_token = response_data['access_token']
    except KeyError as caught_exc:
        new_exc = google_auth_exceptions.RefreshError('No access token in response.', response_data)
        six.raise_from(new_exc, caught_exc)
    refresh_token = response_data.get('refresh_token', refresh_token)
    expiry = google_auth_client._parse_expiry(response_data)
    return (access_token, refresh_token, expiry, response_data)