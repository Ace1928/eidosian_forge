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
def UpdateConsumerPolicyV2Alpha(consumerpolicy, name, force=False, validateonly=False):
    """Make API call to update a consumer policy.

  Args:
    consumerpolicy: The consumer policy to update.
    name: The resource name of the policy. Currently supported format
      '{resource_type}/{resource_name}/consumerPolicies/default. For example,
      'projects/100/consumerPolicies/default'.
    force: Disable service with usage within last 30 days or disable recently
      enabled service.
    validateonly: If set, validate the request and preview the result but do not
      actually commit it. The default is false.

  Raises:
    exceptions.UpdateConsumerPolicyPermissionDeniedException: when updating a
      consumer policy fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    Updated consumer policy
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageConsumerPoliciesPatchRequest(googleApiServiceusageV2alphaConsumerPolicy=consumerpolicy, name=name, force=force, validateOnly=validateonly)
    try:
        return client.consumerPolicies.Patch(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.UpdateConsumerPolicyPermissionDeniedException)
    except apitools_exceptions.HttpBadRequestError as e:
        log.status.Print('Provide the --force flag if you wish to force disable services.')
        exceptions.ReraiseError(e, exceptions.Error)