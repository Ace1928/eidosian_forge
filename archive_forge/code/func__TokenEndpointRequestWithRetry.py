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
@retry.RetryOnException(max_retrials=1, should_retry_if=_ShouldRetryServerInternalError)
def _TokenEndpointRequestWithRetry(request, token_uri, body):
    """Makes a request to the OAuth 2.0 authorization server's token endpoint.

  Args:
      request: google.auth.transport.Request, A callable used to make HTTP
        requests.
      token_uri: str, The OAuth 2.0 authorizations server's token endpoint URI.
      body: {str: str}, The parameters to send in the request body.

  Returns:
      The JSON-decoded response data.
  """
    body = urllib.parse.urlencode(body)
    headers = {'content-type': google_auth_client._URLENCODED_CONTENT_TYPE}
    response = request(method='POST', url=token_uri, headers=headers, body=body)
    response_body = six.ensure_text(response.data)
    if response.status != http_client.OK:
        _HandleErrorResponse(response_body)
    response_data = json.loads(response_body)
    return response_data