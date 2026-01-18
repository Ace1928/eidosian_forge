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
def _CreateContainerManifest(args, instance_name, container_mount_disk_enabled=False, container_mount_disk=None):
    """Create container manifest from argument namespace and instance name."""
    container = {'image': args.container_image, 'name': instance_name}
    if args.container_command is not None:
        container['command'] = [args.container_command]
    if args.container_arg is not None:
        container['args'] = args.container_arg
    container['stdin'] = args.container_stdin
    container['tty'] = args.container_tty
    container['securityContext'] = {'privileged': args.container_privileged}
    env_vars = _ReadDictionary(args.container_env_file)
    for env_var_dict in args.container_env or []:
        for env, val in six.iteritems(env_var_dict):
            env_vars[env] = val
    if env_vars:
        container['env'] = [{'name': env, 'value': val} for env, val in six.iteritems(env_vars)]
    volumes = []
    volume_mounts = []
    for idx, volume in enumerate(args.container_mount_host_path or []):
        volumes.append({'name': _GetHostPathDiskName(idx), 'hostPath': {'path': volume['host-path']}})
        volume_mounts.append({'name': _GetHostPathDiskName(idx), 'mountPath': volume['mount-path'], 'readOnly': volume.get('mode', _DEFAULT_MODE).isReadOnly()})
    for idx, tmpfs in enumerate(args.container_mount_tmpfs or []):
        volumes.append({'name': _GetTmpfsDiskName(idx), 'emptyDir': {'medium': 'Memory'}})
        volume_mounts.append({'name': _GetTmpfsDiskName(idx), 'mountPath': tmpfs['mount-path']})
    if container_mount_disk_enabled:
        container_mount_disk = container_mount_disk or []
        disks = (args.disk or []) + (args.create_disk or [])
        _AddMountedDisksToManifest(container_mount_disk, volumes, volume_mounts, disks=disks)
    container['volumeMounts'] = volume_mounts
    manifest = {'spec': {'containers': [container], 'volumes': volumes, 'restartPolicy': RESTART_POLICY_API[args.container_restart_policy]}}
    return manifest