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
def _AddToApisMap(api_name, api_version, api_def):
    """Adds the APIDef specified by the given arguments to the APIs map.

  This method should only be used for runtime patching of the APIs map.
  Additions to the map should ensure that there is only one and only one default
  version for each API.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    api_def: APIDef for the API version.
  """
    api_name, _ = apis_internal._GetApiNameAndAlias(api_name)
    api_versions = apis_map.MAP.get(api_name, {})
    api_def.default_version = not api_versions
    api_versions[api_version] = api_def
    apis_map.MAP[api_name] = api_versions