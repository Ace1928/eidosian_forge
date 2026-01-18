from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
def SetOrgPolicyRequest(args, policy):
    """Constructs a resource-dependent SetOrgPolicyRequest.

  Args:
    args: Command line arguments.
    policy: OrgPolicy for resource-dependent SetOrgPolicyRequest.

  Returns:
    Resource-dependent SetOrgPolicyRequest.
  """
    messages = org_policies.OrgPoliciesMessages()
    resource_id = GetResource(args)
    request = messages.SetOrgPolicyRequest(policy=policy)
    if args.project:
        return messages.CloudresourcemanagerProjectsSetOrgPolicyRequest(projectsId=resource_id, setOrgPolicyRequest=request)
    elif args.organization:
        return messages.CloudresourcemanagerOrganizationsSetOrgPolicyRequest(organizationsId=resource_id, setOrgPolicyRequest=request)
    elif args.folder:
        return messages.CloudresourcemanagerFoldersSetOrgPolicyRequest(foldersId=resource_id, setOrgPolicyRequest=request)
    return None