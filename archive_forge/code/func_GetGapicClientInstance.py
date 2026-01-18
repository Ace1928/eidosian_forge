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
def GetGapicClientInstance(api_name, api_version, address_override_func=None, transport=apis_util.GapicTransport.GRPC, attempt_direct_path=False):
    """Returns an instance of the GAPIC API client specified in the args.

  Args:
    api_name: str, The API name (or the command surface name, if different).
    api_version: str, The version of the API.
    address_override_func: function, function to call to override the client
      host. It takes a single argument which is the original host.
    transport: apis_util.GapicTransport, The transport to be used by the client.
    attempt_direct_path: bool, True if we want to attempt direct path gRPC where
      possible.

  Raises:
    GapicRestUnsupportedError: If transport is REST.

  Returns:
    An instance of the specified GAPIC API client.
  """
    from googlecloudsdk.core import gapic_util
    if transport == apis_util.GapicTransport.REST:
        raise GapicRestUnsupportedError()
    credentials = gapic_util.GetGapicCredentials()
    return apis_internal._GetGapicClientInstance(api_name, api_version, credentials, address_override_func=address_override_func, transport_choice=transport, attempt_direct_path=attempt_direct_path)