from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _IsAllowed(resource_registry, project_id, policy, errors_to_propagate):
    """Decides if project is allowed within policy."""
    if policy.allValues is policy.AllValuesValueValuesEnum.ALLOW:
        return True
    elif policy.allValues is policy.AllValuesValueValuesEnum.DENY:
        return False
    is_allowed = False
    if not policy.allowedValues:
        is_allowed = True
    try:
        for project_record in policy.allowedValues:
            resource_registry.ParseRelativeName(project_record, 'compute.projects')
    except resources.InvalidResourceException as e:
        errors_to_propagate.append(e)
        is_allowed = True
    else:
        if resource_registry.Parse(project_id, collection='compute.projects').RelativeName() in policy.allowedValues:
            is_allowed = True
    is_denied = False
    try:
        for project_record in policy.deniedValues:
            resource_registry.ParseRelativeName(project_record, 'compute.projects')
    except resources.InvalidResourceException as e:
        is_denied = False
        errors_to_propagate.append(e)
    else:
        if resource_registry.Parse(project_id, collection='compute.projects').RelativeName() in policy.deniedValues:
            is_denied = True
    return is_allowed and (not is_denied)