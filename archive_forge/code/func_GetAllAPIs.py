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
def GetAllAPIs():
    """Gets all registered APIs.

  Returns:
    [API], A list of API definitions.
  """
    all_apis = []
    for api_name, versions in six.iteritems(apis_map.MAP):
        for api_version, _ in six.iteritems(versions):
            all_apis.append(GetAPI(api_name, api_version))
    return all_apis