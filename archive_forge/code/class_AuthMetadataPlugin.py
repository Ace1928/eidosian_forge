from __future__ import absolute_import
import logging
import os
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account
class AuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """A `gRPC AuthMetadataPlugin`_ that inserts the credentials into each
    request.

    .. _gRPC AuthMetadataPlugin:
        http://www.grpc.io/grpc/python/grpc.html#grpc.AuthMetadataPlugin

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to requests.
        request (google.auth.transport.Request): A HTTP transport request
            object used to refresh credentials as needed.
        default_host (Optional[str]): A host like "pubsub.googleapis.com".
            This is used when a self-signed JWT is created from service
            account credentials.
    """

    def __init__(self, credentials, request, default_host=None):
        super(AuthMetadataPlugin, self).__init__()
        self._credentials = credentials
        self._request = request
        self._default_host = default_host

    def _get_authorization_headers(self, context):
        """Gets the authorization headers for a request.

        Returns:
            Sequence[Tuple[str, str]]: A list of request headers (key, value)
                to add to the request.
        """
        headers = {}
        if isinstance(self._credentials, service_account.Credentials):
            self._credentials._create_self_signed_jwt('https://{}/'.format(self._default_host) if self._default_host else None)
        self._credentials.before_request(self._request, context.method_name, context.service_url, headers)
        return list(six.iteritems(headers))

    def __call__(self, context, callback):
        """Passes authorization metadata into the given callback.

        Args:
            context (grpc.AuthMetadataContext): The RPC context.
            callback (grpc.AuthMetadataPluginCallback): The callback that will
                be invoked to pass in the authorization metadata.
        """
        callback(self._get_authorization_headers(context), None)