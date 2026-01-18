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
def _GetClientClassFromDef(api_def):
    """Returns the apitools client class for the API definition specified in args.

  Args:
    api_def: apis_map.APIDef, The definition of the API.

  Returns:
    base_api.BaseApiClient, Client class for the specified API.
  """
    client_full_classpath = api_def.apitools.client_full_classpath
    module_path, client_class_name = client_full_classpath.rsplit('.', 1)
    module_obj = __import__(module_path, fromlist=[client_class_name])
    return getattr(module_obj, client_class_name)