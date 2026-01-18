from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.oauth2 import utils as oauth2_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from six.moves import http_client
from six.moves import urllib
class IntrospectionClient(oauth2_utils.OAuthClientAuthHandler):
    """Implements the OAuth 2.0 token introspection spec.

  This is based on https://tools.ietf.org/html/rfc7662.
  The implementation supports 3 types of client authentication when calling
  the endpoints: no authentication, basic header authentication and POST body
  authentication.
  """

    def __init__(self, token_introspect_endpoint, client_authentication=None):
        """Initializes an OAuth introspection client instance.

    Args:
      token_introspect_endpoint (str): The token introspection endpoint.
      client_authentication (Optional[oauth2_utils.ClientAuthentication]): The
        optional OAuth client authentication credentials if available.
    """
        super(IntrospectionClient, self).__init__(client_authentication)
        self._token_introspect_endpoint = token_introspect_endpoint

    def introspect(self, request, token, token_type_hint=_ACCESS_TOKEN_TYPE):
        """Returns the meta-information associated with an OAuth token.

    Args:
      request (google.auth.transport.Request): A callable that makes HTTP
        requests.
      token (str): The OAuth token whose meta-information are to be returned.
      token_type_hint (Optional[str]): The optional token type. The default is
        access_token.

    Returns:
      Mapping: The active token meta-information returned by the introspection
        endpoint.

    Raises:
      InactiveCredentialsError: If the credentials are invalid or expired.
      TokenIntrospectionError: If an error is encountered while calling the
        token introspection endpoint.
    """
        headers = _URLENCODED_HEADERS.copy()
        request_body = {'token': token, 'token_type_hint': token_type_hint}
        self.apply_client_authentication_options(headers, request_body)
        response = request(url=self._token_introspect_endpoint, method='POST', headers=headers, body=urllib.parse.urlencode(request_body).encode('utf-8'))
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        if response.status != http_client.OK:
            raise TokenIntrospectionError(response_body)
        response_data = json.loads(response_body)
        if response_data.get('active'):
            return response_data
        else:
            raise InactiveCredentialsError(response_body)