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
def AddEnableRule(services: List[str], project: str, consumer_policy_name: str='default', folder: str=None, organization: str=None, validate_only: bool=False):
    """Make API call to enable a specific service.

  Args:
    services: The identifier of the service to enable, for example
      'serviceusage.googleapis.com'.
    project: The project for which to enable the service.
    consumer_policy_name: Name of consumer policy. The default name is
      "default".
    folder: The folder for which to enable the service.
    organization: The organization for which to enable the service.
    validate_only: If True, the action will be validated and result will be
      preview but not exceuted.

  Raises:
    exceptions.EnableServicePermissionDeniedException: when enabling API fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The result of the operation
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    resource_name = _PROJECT_RESOURCE % project
    if folder:
        resource_name = _FOLDER_RESOURCE % folder
    if organization:
        resource_name = _ORGANIZATION_RESOURCE % organization
    policy_name = resource_name + _CONSUMER_POLICY_DEFAULT % consumer_policy_name
    try:
        policy = GetConsumerPolicyV2Alpha(policy_name)
        services_to_enabled = set()
        for service in services:
            services_to_enabled.add(_SERVICE_RESOURCE % service)
            request = messages.ServiceusageServicesGroupsDescendantServicesListRequest(parent='{}/{}'.format(resource_name, _SERVICE_RESOURCE % service + _DEPENDENCY_GROUP))
            try:
                list_descendant_services = client.services_groups_descendantServices.List(request).services
                for member in list_descendant_services:
                    services_to_enabled.add(member.serviceName)
            except apitools_exceptions.HttpNotFoundError:
                continue
        if policy.enableRules:
            policy.enableRules[0].services.extend(list(services_to_enabled))
        else:
            policy.enableRules.append(messages.GoogleApiServiceusageV2alphaEnableRule(services=list(services_to_enabled)))
        if validate_only:
            _GetServices(policy, policy_name, force=False, validate_only=True)
            return
        else:
            return UpdateConsumerPolicyV2Alpha(policy, policy_name)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.EnableServicePermissionDeniedException)