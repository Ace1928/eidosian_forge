import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def access_token_call_credentials(access_token):
    """Construct CallCredentials from an access token.

    Args:
      access_token: A string to place directly in the http request
        authorization header, for example
        "authorization: Bearer <access_token>".

    Returns:
      A CallCredentials.
    """
    from grpc import _auth
    from grpc import _plugin_wrapping
    return _plugin_wrapping.metadata_plugin_call_credentials(_auth.AccessTokenAuthMetadataPlugin(access_token), None)