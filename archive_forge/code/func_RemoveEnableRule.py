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
def RemoveEnableRule(project: str, service: str, consumer_policy_name: str='default', force: bool=False, folder: str=None, organization: str=None, validate_only: bool=False):
    """Make API call to disable a specific service.

  Args:
    project: The project for which to disable the service.
    service: The identifier of the service to disable, for example
      'serviceusage.googleapis.com'.
    consumer_policy_name: Name of consumer policy. The default name is
      "default".
    force: Disable service with usage within last 30 days or disable recently
      enabled service or disable the service even if there are enabled services
      which depend on it. This also disables the services which depend on the
      service to be disabled.
    folder: The folder for which to disable the service.
    organization: The organization for which to disable the service.
    validate_only: If True, the action will be validated and result will be
      preview but not exceuted.`

  Raises:
    exceptions.EnableServicePermissionDeniedException: when disabling API fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The result of the operation
  """
    resource_name = _PROJECT_RESOURCE % project
    if folder:
        resource_name = _FOLDER_RESOURCE % folder
    if organization:
        resource_name = _ORGANIZATION_RESOURCE % organization
    policy_name = resource_name + _CONSUMER_POLICY_DEFAULT % consumer_policy_name
    try:
        current_policy = GetConsumerPolicyV2Alpha(policy_name)
        ancestor_groups = ListAncestorGroups(resource_name, _SERVICE_RESOURCE % service)
        if not force:
            enabled = set()
            for enable_rule in current_policy.enableRules:
                enabled.update(enable_rule.services)
            enabled_dependents = set()
            for ancestor_group in ancestor_groups:
                service_name = '/'.join(str.split(ancestor_group.groupName, '/')[:2])
                if service_name in enabled:
                    enabled_dependents.add(service_name)
            if enabled_dependents:
                enabled_dependents = ','.join(enabled_dependents)
                raise exceptions.ConfigError('The service ' + service + ' is depended on by the following active service(s) ' + enabled_dependents + ' . Provide the --force flag if you wish to force disable services.')
        to_remove = {_SERVICE_RESOURCE % service}
        for ancestor_group in ancestor_groups:
            to_remove.add('/'.join(str.split(ancestor_group.groupName, '/')[:2]))
        updated_consumer_poicy = copy.deepcopy(current_policy)
        updated_consumer_poicy.enableRules.clear()
        for enable_rule in current_policy.enableRules:
            rule = copy.deepcopy(enable_rule)
            for service_name in enable_rule.services:
                if service_name in to_remove:
                    rule.services.remove(service_name)
            if rule.services:
                updated_consumer_poicy.enableRules.append(rule)
        if validate_only:
            _GetServices(updated_consumer_poicy, policy_name, force=force, validate_only=True)
            return
        else:
            return UpdateConsumerPolicyV2Alpha(updated_consumer_poicy, policy_name, force=force)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.EnableServicePermissionDeniedException)
    except apitools_exceptions.HttpBadRequestError as e:
        log.status.Print('Provide the --force flag if you wish to force disable services.')
        exceptions.ReraiseError(e, exceptions.Error)