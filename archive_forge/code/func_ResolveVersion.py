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
def ResolveVersion(api_name, api_version=None):
    """Resolves the version for an API based on the APIs map and API overrides.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The API version.

  Raises:
    apis_internal.UnknownAPIError: If api_name does not exist in the APIs map.

  Returns:
    str, The resolved version.
  """
    api_name, api_name_alias = apis_internal._GetApiNameAndAlias(api_name)
    if api_name not in apis_map.MAP:
        raise apis_util.UnknownAPIError(api_name)
    version_overrides = properties.VALUES.api_client_overrides.AllValues()
    api_version_override = None
    if api_version:
        api_version_override = version_overrides.get('{}/{}'.format(api_name_alias, api_version), None)
    if not api_version_override:
        api_version_override = version_overrides.get(api_name_alias, api_version)
    return api_version_override or apis_internal._GetDefaultVersion(api_name)