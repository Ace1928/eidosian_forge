from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.generated_clients.apis import apis_map
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.parse import urlparse
def _GetClientInstance(api_name, api_version, no_http=False, http_client=None, check_response_func=None, http_timeout_sec=None):
    """Returns an instance of the API client specified in the args.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    no_http: bool, True to not create an http object for this client.
    http_client: bring your own http client to use. Incompatible with
      no_http=True.
    check_response_func: error handling callback to give to apitools.
    http_timeout_sec: int, seconds of http timeout to set, defaults if None.

  Returns:
    base_api.BaseApiClient, An instance of the specified API client.
  """
    if no_http:
        assert http_client is None
    elif http_client is None:
        from googlecloudsdk.core.credentials import transports
        http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, timeout=http_timeout_sec if http_timeout_sec else 'unset')
    client_class = _GetClientClass(api_name, api_version)
    client_instance = client_class(url=_GetEffectiveApiEndpoint(api_name, api_version, client_class), get_credentials=False, http=http_client)
    if check_response_func is not None:
        client_instance.check_response_func = check_response_func
    api_key = properties.VALUES.core.api_key.Get()
    if api_key:
        client_instance.AddGlobalParam('key', api_key)
        header = 'X-Google-Project-Override'
        client_instance.additional_http_headers[header] = 'apikey'
    return client_instance