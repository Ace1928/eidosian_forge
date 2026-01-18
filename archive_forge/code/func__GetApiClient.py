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
def _GetApiClient(api_name, api_version):
    """Gets the repesctive api client for the api."""
    api_def = apis_internal.GetApiDef(api_name, api_version)
    if api_def.apitools:
        client = apis.GetClientInstance(api_name, api_version, no_http=True)
    else:
        client = apis.GetGapicClientInstance(api_name, api_version)
    return client