from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
def MakeRestClient(client_class, credentials, address_override_func=None, mtls_enabled=False):
    """Instantiates a gapic REST client with gcloud defaults and configuration.

  Args:
    client_class: a gapic client class.
    credentials: google.auth.credentials.Credentials, the credentials to use.
    address_override_func: function, function to call to override the client
      host. It takes a single argument which is the original host.
    mtls_enabled: bool, True if mTLS is enabled for this client. _

  Returns:
    A gapic API client.
  """
    transport_class = client_class.get_transport_class('rest')
    address = client_class.DEFAULT_MTLS_ENDPOINT if mtls_enabled else client_class.DEFAULT_ENDPOINT
    if address_override_func:
        address = address_override_func(address)
    return client_class(transport=transport_class(host=address, credentials=credentials))