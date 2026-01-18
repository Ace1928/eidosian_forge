from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _preprocess_mounts(module, values):
    last = dict()

    def check_collision(t, name):
        if t in last:
            if name == last[t]:
                module.fail_json(msg='The mount point "{0}" appears twice in the {1} option'.format(t, name))
            else:
                module.fail_json(msg='The mount point "{0}" appears both in the {1} and {2} option'.format(t, name, last[t]))
        last[t] = name
    if 'mounts' in values:
        mounts = []
        for mount in values['mounts']:
            target = mount['target']
            mount_type = mount['type']
            check_collision(target, 'mounts')
            mount_dict = dict(mount)
            if mount['source'] is None and mount_type not in ('tmpfs', 'volume'):
                module.fail_json(msg='source must be specified for mount "{0}" of type "{1}"'.format(target, mount_type))
            for option, req_mount_type in _MOUNT_OPTION_TYPES.items():
                if mount[option] is not None and mount_type != req_mount_type:
                    module.fail_json(msg='{0} cannot be specified for mount "{1}" of type "{2}" (needs type "{3}")'.format(option, target, mount_type, req_mount_type))
            volume_options = mount_dict.pop('volume_options')
            if mount_dict['volume_driver'] and volume_options:
                mount_dict['volume_options'] = clean_dict_booleans_for_docker_api(volume_options)
            if mount_dict['labels']:
                mount_dict['labels'] = clean_dict_booleans_for_docker_api(mount_dict['labels'])
            if mount_dict['tmpfs_size'] is not None:
                try:
                    mount_dict['tmpfs_size'] = human_to_bytes(mount_dict['tmpfs_size'])
                except ValueError as exc:
                    module.fail_json(msg='Failed to convert tmpfs_size of mount "{0}" to bytes: {1}'.format(target, to_native(exc)))
            if mount_dict['tmpfs_mode'] is not None:
                try:
                    mount_dict['tmpfs_mode'] = int(mount_dict['tmpfs_mode'], 8)
                except Exception as dummy:
                    module.fail_json(msg='tmp_fs mode of mount "{0}" is not an octal string!'.format(target))
            mounts.append(omit_none_from_dict(mount_dict))
        values['mounts'] = mounts
    if 'volumes' in values:
        new_vols = []
        for vol in values['volumes']:
            parts = vol.split(':')
            if ':' in vol:
                if len(parts) == 3:
                    host, container, mode = parts
                    if not _is_volume_permissions(mode):
                        module.fail_json(msg='Found invalid volumes mode: {0}'.format(mode))
                    if re.match('[.~]', host):
                        host = os.path.abspath(os.path.expanduser(host))
                    check_collision(container, 'volumes')
                    new_vols.append('%s:%s:%s' % (host, container, mode))
                    continue
                elif len(parts) == 2:
                    if not _is_volume_permissions(parts[1]) and re.match('[.~]', parts[0]):
                        host = os.path.abspath(os.path.expanduser(parts[0]))
                        check_collision(parts[1], 'volumes')
                        new_vols.append('%s:%s:rw' % (host, parts[1]))
                        continue
            check_collision(parts[min(1, len(parts) - 1)], 'volumes')
            new_vols.append(vol)
        values['volumes'] = new_vols
        new_binds = []
        for vol in new_vols:
            host = None
            if ':' in vol:
                parts = vol.split(':')
                if len(parts) == 3:
                    host, container, mode = parts
                    if not _is_volume_permissions(mode):
                        module.fail_json(msg='Found invalid volumes mode: {0}'.format(mode))
                elif len(parts) == 2:
                    if not _is_volume_permissions(parts[1]):
                        host, container, mode = parts + ['rw']
            if host is not None:
                new_binds.append('%s:%s:%s' % (host, container, mode))
        values['volume_binds'] = new_binds
    return values