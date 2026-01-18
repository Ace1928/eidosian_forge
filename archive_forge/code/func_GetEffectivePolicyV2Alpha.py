import collections
import copy
import enum
import sys
from typing import List
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
def GetEffectivePolicyV2Alpha(name: str, view: str='BASIC'):
    """Make API call to get a effective policy.

  Args:
    name: The name of the effective policy.Currently supported format
      '{resource_type}/{resource_name}/effectivePolicy'. For example,
      'projects/100/effectivePolicy'.
    view: The view of the effective policy to use. The default view is 'BASIC'.

  Raises:
    exceptions.GetEffectiverPolicyPermissionDeniedException: when getting a
      effective policy fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The Effective Policy
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    if view == 'BASIC':
        view_type = messages.ServiceusageGetEffectivePolicyRequest.ViewValueValuesEnum.EFFECTIVE_POLICY_VIEW_BASIC
    else:
        view_type = messages.ServiceusageGetEffectivePolicyRequest.ViewValueValuesEnum.EFFECTIVE_POLICY_VIEW_FULL
    request = messages.ServiceusageGetEffectivePolicyRequest(name=name, view=view_type)
    try:
        return client.v2alpha.GetEffectivePolicy(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.GetEffectiverPolicyPermissionDeniedException)