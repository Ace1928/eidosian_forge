from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instance_groups.flags import AutoDeleteFlag
from googlecloudsdk.command_lib.compute.instance_groups.flags import STATEFUL_IP_DEFAULT_INTERFACE_NAME
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def CreatePerInstanceConfigMessage(holder, instance_ref, stateful_disks, stateful_metadata, disk_getter=None):
    """Create per-instance config message from the given stateful disks and metadata."""
    if not disk_getter:
        disk_getter = instance_disk_getter.InstanceDiskGetter(instance_ref=instance_ref, holder=holder)
    messages = holder.client.messages
    preserved_state_disks = []
    for stateful_disk in stateful_disks or []:
        preserved_state_disks.append(MakePreservedStateDiskEntry(messages, stateful_disk, disk_getter))
    preserved_state_metadata = []
    for metadata_key, metadata_value in sorted(six.iteritems(stateful_metadata)):
        preserved_state_metadata.append(MakePreservedStateMetadataEntry(messages, key=metadata_key, value=metadata_value))
    per_instance_config = messages.PerInstanceConfig(name=path_simplifier.Name(six.text_type(instance_ref)))
    per_instance_config.preservedState = messages.PreservedState(disks=messages.PreservedState.DisksValue(additionalProperties=preserved_state_disks), metadata=messages.PreservedState.MetadataValue(additionalProperties=preserved_state_metadata))
    return per_instance_config