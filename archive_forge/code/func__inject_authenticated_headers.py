import abc
import base64
import enum
import json
import six
from google.auth import exceptions
def _inject_authenticated_headers(self, headers, bearer_token=None):
    if bearer_token is not None:
        headers['Authorization'] = 'Bearer %s' % bearer_token
    elif self._client_authentication is not None and self._client_authentication.client_auth_type is ClientAuthType.basic:
        username = self._client_authentication.client_id
        password = self._client_authentication.client_secret or ''
        credentials = base64.b64encode(('%s:%s' % (username, password)).encode()).decode()
        headers['Authorization'] = 'Basic %s' % credentials