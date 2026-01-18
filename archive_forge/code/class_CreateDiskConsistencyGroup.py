from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CreateDiskConsistencyGroup(base.CreateCommand):
    """Create a Compute Engine Disk Consistency Group resource policy."""

    @staticmethod
    def Args(parser):
        CreateDiskConsistencyGroup.resource_policy_arg = flags.MakeResourcePolicyArg()
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        policy_ref = self.resource_policy_arg.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        messages = holder.client.messages
        resource_policy = util.MakeDiskConsistencyGroupPolicy(policy_ref, args, messages)
        create_request = messages.ComputeResourcePoliciesInsertRequest(resourcePolicy=resource_policy, project=policy_ref.project, region=policy_ref.region)
        service = holder.client.apitools_client.resourcePolicies
        return client.MakeRequests([(service, 'Insert', create_request)])[0]