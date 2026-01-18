from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _set_values_mounts(module, data, api_version, options, values):
    if 'mounts' in values:
        if 'HostConfig' not in data:
            data['HostConfig'] = {}
        mounts = []
        for mount in values['mounts']:
            mount_type = mount.get('type')
            mount_res = {'Target': mount.get('target'), 'Source': mount.get('source'), 'Type': mount_type, 'ReadOnly': mount.get('read_only')}
            if 'consistency' in mount:
                mount_res['Consistency'] = mount['consistency']
            if mount_type == 'bind':
                if 'propagation' in mount:
                    mount_res['BindOptions'] = {'Propagation': mount['propagation']}
            if mount_type == 'volume':
                volume_opts = {}
                if mount.get('no_copy'):
                    volume_opts['NoCopy'] = True
                if mount.get('labels'):
                    volume_opts['Labels'] = mount.get('labels')
                if mount.get('volume_driver'):
                    driver_config = {'Name': mount.get('volume_driver')}
                    if mount.get('volume_options'):
                        driver_config['Options'] = mount.get('volume_options')
                    volume_opts['DriverConfig'] = driver_config
                if volume_opts:
                    mount_res['VolumeOptions'] = volume_opts
            if mount_type == 'tmpfs':
                tmpfs_opts = {}
                if mount.get('tmpfs_mode'):
                    tmpfs_opts['Mode'] = mount.get('tmpfs_mode')
                if mount.get('tmpfs_size'):
                    tmpfs_opts['SizeBytes'] = mount.get('tmpfs_size')
                if tmpfs_opts:
                    mount_res['TmpfsOptions'] = tmpfs_opts
            mounts.append(mount_res)
        data['HostConfig']['Mounts'] = mounts
    if 'volumes' in values:
        volumes = {}
        for volume in values['volumes']:
            if ':' in volume:
                parts = volume.split(':')
                if len(parts) == 3:
                    continue
                if len(parts) == 2:
                    if not _is_volume_permissions(parts[1]):
                        continue
            volumes[volume] = {}
        data['Volumes'] = volumes
    if 'volume_binds' in values:
        if 'HostConfig' not in data:
            data['HostConfig'] = {}
        data['HostConfig']['Binds'] = values['volume_binds']