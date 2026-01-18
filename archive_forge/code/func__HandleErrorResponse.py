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
def _HandleErrorResponse(response_body):
    """"Translates an error response into an exception.

  Args:
      response_body: str, The decoded response data.

  Raises:
      google.auth.exceptions.RefreshError: If the token endpoint returned
          an server internal error.
      ContextAwareAccessDeniedError: if the error was due to a context aware
          access restriction.
      ReauthRequiredError: If reauth is required.
  """
    error_data = json.loads(response_body)
    error_code = error_data.get('error')
    error_subtype = error_data.get('error_subtype')
    if error_code == oauth2client_client.REAUTH_NEEDED_ERROR and (error_subtype == oauth2client_client.REAUTH_NEEDED_ERROR_INVALID_RAPT or error_subtype == oauth2client_client.REAUTH_NEEDED_ERROR_RAPT_REQUIRED):
        raise ReauthRequiredError('reauth is required.')
    try:
        google_auth_client._handle_error_response(error_data, False)
    except google_auth_exceptions.RefreshError as e:
        if context_aware.IsContextAwareAccessDeniedError(e):
            raise ContextAwareAccessDeniedError()
        raise