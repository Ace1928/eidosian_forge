from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
class BtrfsCommands(object):
    """
    Provides access to a subset of the Btrfs command line
    """

    def __init__(self, module):
        self.__module = module
        self.__btrfs = self.__module.get_bin_path('btrfs', required=True)

    def filesystem_show(self):
        command = '%s filesystem show -d' % self.__btrfs
        result = self.__module.run_command(command, check_rc=True)
        stdout = [x.strip() for x in result[1].splitlines()]
        filesystems = []
        current = None
        for line in stdout:
            if line.startswith('Label'):
                current = self.__parse_filesystem(line)
                filesystems.append(current)
            elif line.startswith('devid'):
                current['devices'].append(self.__parse_filesystem_device(line))
        return filesystems

    def __parse_filesystem(self, line):
        label = re.sub('\\s*uuid:.*$', '', re.sub('^Label:\\s*', '', line))
        id = re.sub('^.*uuid:\\s*', '', line)
        filesystem = {}
        filesystem['label'] = label.strip("'") if label != 'none' else None
        filesystem['uuid'] = id
        filesystem['devices'] = []
        filesystem['mountpoints'] = []
        filesystem['subvolumes'] = []
        filesystem['default_subvolid'] = None
        return filesystem

    def __parse_filesystem_device(self, line):
        return re.sub('^.*path\\s', '', line)

    def subvolumes_list(self, filesystem_path):
        command = '%s subvolume list -tap %s' % (self.__btrfs, filesystem_path)
        result = self.__module.run_command(command, check_rc=True)
        stdout = [x.split('\t') for x in result[1].splitlines()]
        subvolumes = [{'id': 5, 'parent': None, 'path': '/'}]
        if len(stdout) > 2:
            subvolumes.extend([self.__parse_subvolume_list_record(x) for x in stdout[2:]])
        return subvolumes

    def __parse_subvolume_list_record(self, item):
        return {'id': int(item[0]), 'parent': int(item[2]), 'path': normalize_subvolume_path(item[5])}

    def subvolume_get_default(self, filesystem_path):
        command = [self.__btrfs, 'subvolume', 'get-default', to_bytes(filesystem_path)]
        result = self.__module.run_command(command, check_rc=True)
        return int(result[1].strip().split()[1])

    def subvolume_set_default(self, filesystem_path, subvolume_id):
        command = [self.__btrfs, 'subvolume', 'set-default', str(subvolume_id), to_bytes(filesystem_path)]
        result = self.__module.run_command(command, check_rc=True)

    def subvolume_create(self, subvolume_path):
        command = [self.__btrfs, 'subvolume', 'create', to_bytes(subvolume_path)]
        result = self.__module.run_command(command, check_rc=True)

    def subvolume_snapshot(self, snapshot_source, snapshot_destination):
        command = [self.__btrfs, 'subvolume', 'snapshot', to_bytes(snapshot_source), to_bytes(snapshot_destination)]
        result = self.__module.run_command(command, check_rc=True)

    def subvolume_delete(self, subvolume_path):
        command = [self.__btrfs, 'subvolume', 'delete', to_bytes(subvolume_path)]
        result = self.__module.run_command(command, check_rc=True)