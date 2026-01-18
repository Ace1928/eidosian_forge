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
def DisableApiCall(project, service, force=False):
    """Make API call to disable a specific service.

  Args:
    project: The project for which to enable the service.
    service: The identifier of the service to disable, for example
      'serviceusage.googleapis.com'.
    force: disable the service even if there are enabled services which depend
      on it. This also disables the services which depend on the service to be
      disabled.

  Raises:
    exceptions.EnableServicePermissionDeniedException: when disabling API fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The result of the operation
  """
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    check = messages.DisableServiceRequest.CheckIfServiceHasUsageValueValuesEnum.CHECK
    if force:
        check = messages.DisableServiceRequest.CheckIfServiceHasUsageValueValuesEnum.SKIP
    request = messages.ServiceusageServicesDisableRequest(name=_PROJECT_SERVICE_RESOURCE % (project, service), disableServiceRequest=messages.DisableServiceRequest(disableDependentServices=force, checkIfServiceHasUsage=check))
    try:
        return client.services.Disable(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.EnableServicePermissionDeniedException)
    except apitools_exceptions.HttpBadRequestError as e:
        log.status.Print('Provide the --force flag if you wish to force disable services.')
        exceptions.ReraiseError(e, exceptions.Error)