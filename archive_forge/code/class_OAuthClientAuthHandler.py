import abc
import base64
import enum
import json
import six
from google.auth import exceptions
@six.add_metaclass(abc.ABCMeta)
class OAuthClientAuthHandler(object):
    """Abstract class for handling client authentication in OAuth-based
    operations.
    """

    def __init__(self, client_authentication=None):
        """Instantiates an OAuth client authentication handler.

        Args:
            client_authentication (Optional[google.oauth2.utils.ClientAuthentication]):
                The OAuth client authentication credentials if available.
        """
        super(OAuthClientAuthHandler, self).__init__()
        self._client_authentication = client_authentication

    def apply_client_authentication_options(self, headers, request_body=None, bearer_token=None):
        """Applies client authentication on the OAuth request's headers or POST
        body.

        Args:
            headers (Mapping[str, str]): The HTTP request header.
            request_body (Optional[Mapping[str, str]]): The HTTP request body
                dictionary. For requests that do not support request body, this
                is None and will be ignored.
            bearer_token (Optional[str]): The optional bearer token.
        """
        self._inject_authenticated_headers(headers, bearer_token)
        if bearer_token is None:
            self._inject_authenticated_request_body(request_body)

    def _inject_authenticated_headers(self, headers, bearer_token=None):
        if bearer_token is not None:
            headers['Authorization'] = 'Bearer %s' % bearer_token
        elif self._client_authentication is not None and self._client_authentication.client_auth_type is ClientAuthType.basic:
            username = self._client_authentication.client_id
            password = self._client_authentication.client_secret or ''
            credentials = base64.b64encode(('%s:%s' % (username, password)).encode()).decode()
            headers['Authorization'] = 'Basic %s' % credentials

    def _inject_authenticated_request_body(self, request_body):
        if self._client_authentication is not None and self._client_authentication.client_auth_type is ClientAuthType.request_body:
            if request_body is None:
                raise exceptions.OAuthError('HTTP request does not support request-body')
            else:
                request_body['client_id'] = self._client_authentication.client_id
                request_body['client_secret'] = self._client_authentication.client_secret or ''