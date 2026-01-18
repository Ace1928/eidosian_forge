from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def TypeProviderClient(version):
    """Return a Type Provider client specially suited for listing types.

  Listing types requires many API calls, some of which may fail due to bad
  user configurations which show up as errors that are retryable. We can
  alleviate some of the latency and usability issues this causes by tuning
  the client.

  Args:
      version: DM API version used for the client.

  Returns:
    A Type Provider API client.
  """
    main_client = apis.GetClientInstance('deploymentmanager', version.id)
    main_client.num_retries = 2
    return main_client.typeProviders