from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakePreservedStateDisksMapEntry(messages, stateful_disk):
    """Make a map entry for disks field in preservedState message."""
    auto_delete_map = {'never': messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum.NEVER, 'on-permanent-instance-deletion': messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum.ON_PERMANENT_INSTANCE_DELETION}
    disk_device = messages.PreservedStatePreservedDisk()
    if 'auto_delete' in stateful_disk:
        disk_device.autoDelete = auto_delete_map[stateful_disk['auto_delete']]
    return messages.PreservedState.DisksValue.AdditionalProperty(key=stateful_disk['device_name'], value=disk_device)