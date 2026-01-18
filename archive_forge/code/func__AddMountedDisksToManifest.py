from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _AddMountedDisksToManifest(container_mount_disk, volumes, volume_mounts, used_names=None, disks=None):
    """Add volume specs from --container-mount-disk."""
    used_names = used_names or []
    disks = disks or []
    idx = 0
    for mount_disk in container_mount_disk:
        while _GetPersistentDiskName(idx) in used_names:
            idx += 1
        device_name = mount_disk.get('name')
        partition = mount_disk.get('partition')

        def _GetMatchingVolume(device_name, partition):
            for volume_spec in volumes:
                pd = volume_spec.get('gcePersistentDisk', {})
                if pd.get('pdName') == device_name and pd.get('partition') == partition:
                    return volume_spec
        repeated = _GetMatchingVolume(device_name, partition)
        if repeated:
            name = repeated['name']
        else:
            name = _GetPersistentDiskName(idx)
            used_names.append(name)
        if not device_name:
            if len(disks) != 1:
                raise calliope_exceptions.InvalidArgumentException('--container-mount-disk', 'Must specify the name of the disk to be mounted unless exactly one disk is attached to the instance.')
            device_name = disks[0].get('name')
            if disks[0].get('device-name', device_name) != device_name:
                raise exceptions.InvalidArgumentException('--container-mount-disk', 'Must not have a device-name that is different from disk name if disk is being attached to the instance and mounted to a container: [{}]'.format(disks[0].get('device-name')))
        volume_mounts.append({'name': name, 'mountPath': mount_disk['mount-path'], 'readOnly': mount_disk.get('mode', _DEFAULT_MODE).isReadOnly()})
        if repeated:
            continue
        volume_spec = {'name': name, 'gcePersistentDisk': {'pdName': device_name, 'fsType': 'ext4'}}
        if partition:
            volume_spec['gcePersistentDisk'].update({'partition': partition})
        volumes.append(volume_spec)
        idx += 1