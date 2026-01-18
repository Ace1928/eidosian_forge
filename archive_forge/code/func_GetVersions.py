from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetVersions(api_name):
    """Return available versions for given api.

  Args:
    api_name: str, The API name (or the command surface name, if different).

  Raises:
    UnknownAPIError: If api_name does not exist in the APIs map.

  Returns:
    list, of version names.
  """
    return apis_internal._GetVersions(api_name)