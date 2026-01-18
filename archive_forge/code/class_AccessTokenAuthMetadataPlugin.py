import inspect
from typing import Any, Optional
import grpc
class AccessTokenAuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """Metadata wrapper for raw access token credentials."""
    _access_token: str

    def __init__(self, access_token: str):
        self._access_token = access_token

    def __call__(self, context: grpc.AuthMetadataContext, callback: grpc.AuthMetadataPluginCallback):
        _sign_request(callback, self._access_token, None)