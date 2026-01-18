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
def _GetGapicClientClass(api_name, api_version, transport_choice=apis_util.GapicTransport.GRPC):
    """Returns the GAPIC client class for the API def specified by the args.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    transport_choice: apis_util.GapicTransport, The transport to be used by the
      client.
  """
    api_def = GetApiDef(api_name, api_version)
    if transport_choice == apis_util.GapicTransport.GRPC_ASYNCIO:
        client_full_classpath = api_def.gapic.async_client_full_classpath
    elif transport_choice == apis_util.GapicTransport.REST:
        client_full_classpath = api_def.gapic.rest_client_full_classpath
    else:
        client_full_classpath = api_def.gapic.client_full_classpath
    module_path, client_class_name = client_full_classpath.rsplit('.', 1)
    module_obj = __import__(module_path, fromlist=[client_class_name])
    return getattr(module_obj, client_class_name)