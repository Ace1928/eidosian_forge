import inspect
from typing import Any, Optional
import grpc
class GoogleCallCredentials(grpc.AuthMetadataPlugin):
    """Metadata wrapper for GoogleCredentials from the oauth2client library."""
    _is_jwt: bool
    _credentials: Any

    def __init__(self, credentials: Any):
        self._credentials = credentials
        self._is_jwt = 'additional_claims' in inspect.getfullargspec(credentials.get_access_token).args

    def __call__(self, context: grpc.AuthMetadataContext, callback: grpc.AuthMetadataPluginCallback):
        try:
            if self._is_jwt:
                access_token = self._credentials.get_access_token(additional_claims={'aud': context.service_url}).access_token
            else:
                access_token = self._credentials.get_access_token().access_token
        except Exception as exception:
            _sign_request(callback, None, exception)
        else:
            _sign_request(callback, access_token, None)