from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_getter
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def _CombinePerInstanceConfigMessage(self, holder, per_instance_config, instance_ref, args):
    update_stateful_disks = args.stateful_disk
    remove_stateful_disks = args.remove_stateful_disks
    update_stateful_metadata = args.stateful_metadata
    remove_stateful_metadata = args.remove_stateful_metadata
    messages = holder.client.messages
    disk_getter = instance_disk_getter.InstanceDiskGetter(instance_ref=instance_ref, holder=holder)
    disks_to_remove_set = set(remove_stateful_disks or [])
    disks_to_update_dict = {update_stateful_disk.get('device-name'): update_stateful_disk for update_stateful_disk in update_stateful_disks or []}
    new_stateful_disks = UpdateGA._UpdateStatefulDisks(messages, per_instance_config, disks_to_update_dict, disks_to_remove_set, disk_getter)
    new_stateful_metadata = UpdateGA._UpdateStatefulMetadata(messages, per_instance_config, update_stateful_metadata, remove_stateful_metadata)
    per_instance_config.preservedState.disks = messages.PreservedState.DisksValue(additionalProperties=new_stateful_disks)
    per_instance_config.preservedState.metadata = messages.PreservedState.MetadataValue(additionalProperties=[instance_configs_messages.MakePreservedStateMetadataEntry(messages, key=key, value=value) for key, value in sorted(six.iteritems(new_stateful_metadata))])
    UpdateGA._PatchStatefulInternalIPs(messages=messages, per_instance_config=per_instance_config, ips_to_update=args.stateful_internal_ip, ips_to_remove=args.remove_stateful_internal_ips)
    UpdateGA._PatchStatefulExternalIPs(messages=messages, per_instance_config=per_instance_config, ips_to_update=args.stateful_external_ip, ips_to_remove=args.remove_stateful_external_ips)
    return per_instance_config