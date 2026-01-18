from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DeletePolicy(base.DeleteCommand):
    """Delete an Organization Policy.

  Deletes an Organization Policy associated with the specified resource.

  ## EXAMPLES

  The following command clears an Organization Policy for constraint
  `serviceuser.services` on project `foo-project`:

    $ {command} serviceuser.services --project=foo-project
  """

    @staticmethod
    def Args(parser):
        flags.AddIdArgToParser(parser)
        flags.AddParentResourceFlagsToParser(parser)

    def Run(self, args):
        service = org_policies_base.OrgPoliciesService(args)
        result = service.ClearOrgPolicy(self.ClearOrgPolicyRequest(args))
        log.DeletedResource(result)

    @staticmethod
    def ClearOrgPolicyRequest(args):
        messages = org_policies.OrgPoliciesMessages()
        resource_id = org_policies_base.GetResource(args)
        request = messages.ClearOrgPolicyRequest(constraint=org_policies.FormatConstraint(args.id))
        if args.project:
            return messages.CloudresourcemanagerProjectsClearOrgPolicyRequest(projectsId=resource_id, clearOrgPolicyRequest=request)
        elif args.organization:
            return messages.CloudresourcemanagerOrganizationsClearOrgPolicyRequest(organizationsId=resource_id, clearOrgPolicyRequest=request)
        elif args.folder:
            return messages.CloudresourcemanagerFoldersClearOrgPolicyRequest(foldersId=resource_id, clearOrgPolicyRequest=request)
        return None