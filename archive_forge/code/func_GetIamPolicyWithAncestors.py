from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from apitools.base.py.exceptions import HttpForbiddenError
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.iam import policies
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import exceptions
from googlecloudsdk.core import resources
import six
def GetIamPolicyWithAncestors(project_id, include_deny, release_track):
    """Get IAM policy for given project and its ancestors.

  Args:
    project_id: project id
    include_deny: boolean that represents if we should show the deny policies in
      addition to the grants
    release_track: which release track, include deny is only supported for ALPHA
      or BETA

  Returns:
    IAM policy for given project and its ancestors
  """
    iam_policies = []
    ancestry = projects_api.GetAncestry(project_id)
    try:
        for resource in ancestry.ancestor:
            resource_type = resource.resourceId.type
            resource_id = resource.resourceId.id
            if resource_type == 'project':
                project_ref = ParseProject(project_id)
                iam_policies.append({'type': 'project', 'id': project_id, 'policy': projects_api.GetIamPolicy(project_ref)})
                if include_deny:
                    deny_policies = policies.ListDenyPolicies(project_id, 'project', release_track)
                    for deny_policy in deny_policies:
                        iam_policies.append({'type': 'project', 'id': project_id, 'policy': deny_policy})
            if resource_type == 'folder':
                iam_policies.append({'type': resource_type, 'id': resource_id, 'policy': folders.GetIamPolicy(resource_id)})
                if include_deny:
                    deny_policies = policies.ListDenyPolicies(resource_id, 'folder', release_track)
                    for deny_policy in deny_policies:
                        iam_policies.append({'type': 'folder', 'id': resource_id, 'policy': deny_policy})
            if resource_type == 'organization':
                iam_policies.append({'type': resource_type, 'id': resource_id, 'policy': organizations.Client().GetIamPolicy(resource_id)})
                if include_deny:
                    deny_policies = policies.ListDenyPolicies(resource_id, 'organization', release_track)
                    for deny_policy in deny_policies:
                        iam_policies.append({'type': 'organization', 'id': resource_id, 'policy': deny_policy})
        return iam_policies
    except HttpForbiddenError:
        raise exceptions.AncestorsIamPolicyAccessDeniedError('User is not permitted to access IAM policy for one or more of the ancestors')