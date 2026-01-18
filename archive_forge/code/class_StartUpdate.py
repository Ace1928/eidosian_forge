from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import rolling_action
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StartUpdate(base.Command):
    """Start restart instances of managed instance group."""

    @staticmethod
    def Args(parser):
        _AddArgs(parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resources = holder.resources
        minimal_action = client.messages.InstanceGroupManagerUpdatePolicy.MinimalActionValueValuesEnum.RESTART
        return client.MakeRequests([rolling_action.CreateRequest(args, client, resources, minimal_action)])