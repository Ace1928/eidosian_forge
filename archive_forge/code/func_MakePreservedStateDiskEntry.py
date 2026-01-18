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
def MakePreservedStateDiskEntry(messages, stateful_disk_data, disk_getter):
    """Prepares disk preserved state entry, combining with params from the instance."""
    if stateful_disk_data.get('source'):
        source = stateful_disk_data.get('source')
        mode = stateful_disk_data.get('mode', 'rw')
    else:
        disk = disk_getter.get_disk(device_name=stateful_disk_data.get('device-name'))
        if disk is None:
            if disk_getter.instance_exists:
                error_message = '[source] is required because the disk with the [device-name]: `{0}` is not yet configured in the instance config'.format(stateful_disk_data.get('device-name'))
            else:
                error_message = '[source] must be given while defining stateful disks in instance configs for new instances'
            raise exceptions.BadArgumentException('stateful_disk', error_message)
        source = disk.source
        mode = stateful_disk_data.get('mode') or disk.mode
    preserved_disk = messages.PreservedStatePreservedDisk(autoDelete=(stateful_disk_data.get('auto-delete') or AutoDeleteFlag.NEVER).GetAutoDeleteEnumValue(messages.PreservedStatePreservedDisk.AutoDeleteValueValuesEnum), source=source, mode=GetMode(messages, mode))
    return messages.PreservedState.DisksValue.AdditionalProperty(key=stateful_disk_data.get('device-name'), value=preserved_disk)