from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from google.auth.transport import requests as google_auth_requests
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
def AuthorizeClient(self, http_client, creds):
    """Returns an http_client authorized with the given credentials."""
    orig_request = http_client.request
    credential_refresh_state = {'attempt': 0}

    def WrappedRequest(method, url, data=None, headers=None, **kwargs):
        wrapped_request = http_client.request
        http_client.request = orig_request
        auth_request = google_auth_requests.Request(http_client)
        creds.before_request(auth_request, method, url, headers)
        http_client.request = wrapped_request
        response = orig_request(method, url, data=data, headers=headers or {}, **kwargs)
        if response.status_code in REFRESH_STATUS_CODES and (not (isinstance(creds, google_auth_external_account.Credentials) and creds.valid)) and (credential_refresh_state['attempt'] < MAX_REFRESH_ATTEMPTS):
            credential_refresh_state['attempt'] += 1
            creds.refresh(requests.GoogleAuthRequest())
            response = orig_request(method, url, data=data, headers=headers or {}, **kwargs)
        return response
    http_client.request = WrappedRequest
    return http_client