from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.args import labels_util
def _GetGracefulShutdownOperationRef(self, args, instance_ref, holder):
    messages = holder.client.messages
    client = holder.client.apitools_client
    instance = client.instances.Get(messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))
    updated_graceful_shutdown_message = instance.scheduling.gracefulShutdown
    if hasattr(args, 'graceful_shutdown') and args.IsSpecified('graceful_shutdown'):
        if updated_graceful_shutdown_message is None:
            updated_graceful_shutdown_message = messages.SchedulingGracefulShutdown()
        updated_graceful_shutdown_message.enabled = args.graceful_shutdown
    if hasattr(args, 'graceful_shutdown_max_duration') and args.IsSpecified('graceful_shutdown_max_duration'):
        if updated_graceful_shutdown_message is None:
            updated_graceful_shutdown_message = messages.SchedulingGracefulShutdown()
        updated_graceful_shutdown_message.maxDuration = messages.Duration(seconds=args.graceful_shutdown_max_duration)
    instance.scheduling.gracefulShutdown = updated_graceful_shutdown_message
    request = messages.ComputeInstancesUpdateRequest(instance=instance_ref.Name(), project=instance_ref.project, zone=instance_ref.zone, instanceResource=instance, minimalAction=messages.ComputeInstancesUpdateRequest.MinimalActionValueValuesEnum.NO_EFFECT, mostDisruptiveAllowedAction=messages.ComputeInstancesUpdateRequest.MostDisruptiveAllowedActionValueValuesEnum.REFRESH)
    operation = client.instances.Update(request)
    return holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')