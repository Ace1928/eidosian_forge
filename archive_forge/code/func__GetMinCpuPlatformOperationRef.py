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
def _GetMinCpuPlatformOperationRef(self, min_cpu_platform, instance_ref, holder):
    client = holder.client.apitools_client
    messages = holder.client.messages
    embedded_request = messages.InstancesSetMinCpuPlatformRequest(minCpuPlatform=min_cpu_platform or None)
    request = messages.ComputeInstancesSetMinCpuPlatformRequest(instance=instance_ref.instance, project=instance_ref.project, instancesSetMinCpuPlatformRequest=embedded_request, zone=instance_ref.zone)
    operation = client.instances.SetMinCpuPlatform(request)
    return holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')