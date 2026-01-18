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
def ListServicesV2Alpha(project, enabled, page_size, limit=sys.maxsize, folder=None, organization=None):
    """Make API call to list services.

  Args:
    project: The project for which to list services.
    enabled: List only enabled services.
    page_size: The page size to list.
    limit: The max number of services to display.
    folder: The folder for which to list services.
    organization: The organization for which to list services.

  Raises:
    exceptions.ListServicesPermissionDeniedException: when listing services
    fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The list of services
  """
    resource_name = _PROJECT_RESOURCE % project
    if folder:
        resource_name = _FOLDER_RESOURCE % folder
    if organization:
        resource_name = _ORGANIZATION_RESOURCE % organization
    services = {}
    parent = []
    try:
        if enabled:
            policy_name = resource_name + _EFFECTIVE_POLICY
            effectivepolicy = GetEffectivePolicyV2Alpha(policy_name)
            for rules in effectivepolicy.enableRules:
                for value in rules.services:
                    if limit == 0:
                        break
                    parent.append(f'{resource_name}/{value}')
                    services[value] = ''
                    limit -= 1
            for value in range(0, len(parent), 20):
                response = BatchGetService(resource_name, parent[value:value + 20])
                for service_state in response.services:
                    service_name = '/'.join(service_state.name.split('/')[2:])
                    services[service_name] = service_state.service.displayName
        else:
            for category_service in ListCategoryServices(resource_name, _GOOGLE_CATEGORY_RESOURCE, page_size, limit):
                services[category_service.service.name] = category_service.service.displayName
        result = []
        service_info = collections.namedtuple('ServiceList', ['name', 'title'])
        for service in services:
            result.append(service_info(name=service, title=services[service]))
        return result
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.EnableServicePermissionDeniedException)