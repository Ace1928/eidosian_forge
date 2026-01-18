import datetime
import errno
import json
import os
import requests
import sys
import time
import webbrowser
import google_auth_oauthlib.flow as auth_flows
import grpc
import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
from tensorboard.uploader import util
from tensorboard.util import tb_logging
class IdTokenAuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """A `gRPC AuthMetadataPlugin` that uses ID tokens.

    This works like the existing `google.auth.transport.grpc.AuthMetadataPlugin`
    except that instead of always using access tokens, it preferentially uses the
    `Credentials.id_token` property if available (and logs an error otherwise).

    See http://www.grpc.io/grpc/python/grpc.html#grpc.AuthMetadataPlugin
    """

    def __init__(self, credentials, request):
        """Constructs an IdTokenAuthMetadataPlugin.

        Args:
          credentials (google.auth.credentials.Credentials): The credentials to
            add to requests.
          request (google.auth.transport.Request): A HTTP transport request object
            used to refresh credentials as needed.
        """
        super().__init__()
        if not isinstance(credentials, google.oauth2.credentials.Credentials):
            raise TypeError('Cannot get ID tokens from credentials type %s' % type(credentials))
        self._credentials = credentials
        self._request = request

    def __call__(self, context, callback):
        """Passes authorization metadata into the given callback.

        Args:
          context (grpc.AuthMetadataContext): The RPC context.
          callback (grpc.AuthMetadataPluginCallback): The callback that will
            be invoked to pass in the authorization metadata.
        """
        headers = {}
        self._credentials.before_request(self._request, context.method_name, context.service_url, headers)
        id_token = getattr(self._credentials, 'id_token', None)
        if id_token:
            self._credentials.apply(headers, token=id_token)
        else:
            logger.error('Failed to find ID token credentials')
        callback(list(headers.items()), None)