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
def _GetClientClass(api_name, api_version):
    """Returns the client class for the API specified in the args.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.

  Returns:
    base_api.BaseApiClient, Client class for the specified API.
  """
    api_def = GetApiDef(api_name, api_version)
    return _GetClientClassFromDef(api_def)