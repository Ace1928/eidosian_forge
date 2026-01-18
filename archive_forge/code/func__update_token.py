import base64
import copy
from datetime import datetime
import json
import six
from six.moves import http_client
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
def _update_token(self, request):
    """Updates credentials with a new access_token representing
        the impersonated account.

        Args:
            request (google.auth.transport.requests.Request): Request object
                to use for refreshing credentials.
        """
    if not self._source_credentials.valid:
        self._source_credentials.refresh(request)
    body = {'delegates': self._delegates, 'scope': self._target_scopes, 'lifetime': str(self._lifetime) + 's'}
    headers = {'Content-Type': 'application/json'}
    self._source_credentials.apply(headers)
    self.token, self.expiry = _make_iam_token_request(request=request, principal=self._target_principal, headers=headers, body=body, iam_endpoint_override=self._iam_endpoint_override)