from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateSnapshotSchedule(base.UpdateCommand):
    """Update a Compute Engine Snapshot Schedule Resource Policy."""

    @staticmethod
    def Args(parser):
        _CommonArgs(parser, compute_api.COMPUTE_GA_API_VERSION)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        messages = holder.client.messages
        policy_ref = flags.MakeResourcePolicyArg().ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        resource_policy = util.MakeDiskSnapshotSchedulePolicyForUpdate(policy_ref, args, messages)
        patch_request = messages.ComputeResourcePoliciesPatchRequest(resourcePolicy=policy_ref.Name(), resourcePolicyResource=resource_policy, project=policy_ref.project, region=policy_ref.region)
        service = holder.client.apitools_client.resourcePolicies
        return client.MakeRequests([(service, 'Patch', patch_request)])