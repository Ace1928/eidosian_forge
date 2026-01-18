from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import code
import site  # pylint: disable=unused-import
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.generated_clients.apis import apis_map
from googlecloudsdk.core import log  # pylint: disable=unused-import
from googlecloudsdk.core import properties  # pylint: disable=unused-import
from googlecloudsdk.core.console import console_io  # pylint: disable=unused-import
from googlecloudsdk.core.util import files  # pylint: disable=unused-import
def LoadApis():
    """Populate the global module namespace with API clients."""
    for api_name in apis_map.MAP:
        globals()[api_name] = apis.GetClientInstance(api_name, apis_internal._GetDefaultVersion(api_name))