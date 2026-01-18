from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetAPICollections(api_name=None, api_version=None):
    """Gets the registered collections for the given API version.

  Args:
    api_name: str, The name of the API or None for all apis.
    api_version: str, The version string of the API or None to use the default
      version.

  Returns:
    [APICollection], A list of the registered collections.
  """
    if api_name:
        all_apis = {api_name: _ValidateAndGetDefaultVersion(api_name, api_version)}
    else:
        all_apis = {x.name: x.version for x in GetAllAPIs() if x.is_default}
    collections = []
    for n, v in six.iteritems(all_apis):
        collections.extend([APICollection(c) for c in apis_internal._GetApiCollections(n, v)])
    return collections