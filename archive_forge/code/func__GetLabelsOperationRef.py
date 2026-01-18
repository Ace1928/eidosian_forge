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
def _GetLabelsOperationRef(self, labels_diff, instance, instance_ref, holder):
    client = holder.client.apitools_client
    messages = holder.client.messages
    labels_update = labels_diff.Apply(messages.InstancesSetLabelsRequest.LabelsValue, instance.labels)
    if labels_update.needs_update:
        request = messages.ComputeInstancesSetLabelsRequest(project=instance_ref.project, instance=instance_ref.instance, zone=instance_ref.zone, instancesSetLabelsRequest=messages.InstancesSetLabelsRequest(labelFingerprint=instance.labelFingerprint, labels=labels_update.labels))
        operation = client.instances.SetLabels(request)
        return holder.resources.Parse(operation.selfLink, collection='compute.zoneOperations')