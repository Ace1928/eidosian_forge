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
def _GetUpdateInstanceSchedulingRef(self, instance_ref, args, holder):
    client = holder.client.apitools_client
    messages = holder.client.messages
    if instance_utils.IsAnySpecified(args, 'node', 'node_affinity_file', 'node_group'):
        affinities = sole_tenancy_util.GetSchedulingNodeAffinityListFromArgs(args, messages)
    elif args.IsSpecified('clear_node_affinities'):
        affinities = []
    else:
        return None
    instance = client.instances.Get(messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))
    instance.scheduling.nodeAffinities = affinities
    request = messages.ComputeInstancesUpdateRequest(instance=instance_ref.Name(), project=instance_ref.project, zone=instance_ref.zone, instanceResource=instance, minimalAction=messages.ComputeInstancesUpdateRequest.MinimalActionValueValuesEnum.NO_EFFECT, mostDisruptiveAllowedAction=messages.ComputeInstancesUpdateRequest.MostDisruptiveAllowedActionValueValuesEnum.REFRESH)
    operation = client.instances.Update(request)
    return holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')