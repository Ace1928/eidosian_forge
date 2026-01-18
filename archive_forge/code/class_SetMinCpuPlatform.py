from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
@base.Deprecate(is_removed=False, warning='This command is deprecated. Use $ gcloud alpha compute instances update --set-min-cpu-platform instead.')
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetMinCpuPlatform(base.UpdateCommand):
    """Set minimum CPU platform for Compute Engine virtual machine instance."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)
        flags.AddMinCpuPlatformArgs(parser, base.ReleaseTrack.ALPHA, required=True)
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        embedded_request = client.messages.InstancesSetMinCpuPlatformRequest(minCpuPlatform=args.min_cpu_platform or None)
        request = client.messages.ComputeInstancesSetMinCpuPlatformRequest(instance=instance_ref.instance, project=instance_ref.project, instancesSetMinCpuPlatformRequest=embedded_request, zone=instance_ref.zone)
        operation = client.apitools_client.instances.SetMinCpuPlatform(request)
        operation_ref = holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')
        if args.async_:
            log.UpdatedResource(operation_ref, kind='gce instance [{0}]'.format(instance_ref.Name()), is_async=True, details='Use [gcloud compute operations describe] command to check the status of this operation.')
            return operation
        operation_poller = poller.Poller(client.apitools_client.instances)
        return waiter.WaitFor(operation_poller, operation_ref, 'Changing minimum CPU platform of instance [{0}]'.format(instance_ref.Name()))