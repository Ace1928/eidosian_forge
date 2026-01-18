from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __execute_create_snapshot(self, operation):
    source_subvolume = self.__filesystem.get_subvolume_by_name(operation['source'])
    source_mounted_path = source_subvolume.get_mounted_path()
    target_mounted_path = self.__filesystem.get_mountpath_as_child(operation['target'])
    self.__btrfs_api.subvolume_snapshot(source_mounted_path, target_mounted_path)
    self.__completed_work.append(operation)