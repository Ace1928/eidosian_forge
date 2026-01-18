from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.api_lib.iam import policies as policies_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import exceptions
from googlecloudsdk.core import resources
def GetAncestorsIamPolicy(folder_id, include_deny, release_track):
    """Gets IAM policies for given folder and its ancestors."""
    policies = []
    resource = GetFolder(folder_id)
    try:
        while resource is not None:
            resource_id = resource.name.split('/')[1]
            policies.append({'type': 'folder', 'id': resource_id, 'policy': GetIamPolicy(resource_id)})
            if include_deny:
                deny_policies = policies_api.ListDenyPolicies(resource_id, 'folder', release_track)
                for deny_policy in deny_policies:
                    policies.append({'type': 'folder', 'id': resource_id, 'policy': deny_policy})
            parent_id = resource.parent.split('/')[1]
            if resource.parent.startswith('folder'):
                resource = GetFolder(parent_id)
            else:
                policies.append({'type': 'organization', 'id': parent_id, 'policy': organizations.Client().GetIamPolicy(parent_id)})
                if include_deny:
                    deny_policies = policies_api.ListDenyPolicies(parent_id, 'organization', release_track)
                    for deny_policy in deny_policies:
                        policies.append({'type': 'organization', 'id': resource_id, 'policy': deny_policy})
                resource = None
    except api_exceptions.HttpForbiddenError:
        raise exceptions.AncestorsIamPolicyAccessDeniedError('User is not permitted to access IAM policy for one or more of the ancestors')
    return policies