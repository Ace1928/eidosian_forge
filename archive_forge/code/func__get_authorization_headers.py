from __future__ import absolute_import
import logging
import os
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account
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