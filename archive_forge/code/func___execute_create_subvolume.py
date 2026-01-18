from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __execute_create_subvolume(self, operation):
    target_mounted_path = self.__filesystem.get_mountpath_as_child(operation['target'])
    if not self.__is_existing_directory_like(target_mounted_path):
        self.__btrfs_api.subvolume_create(target_mounted_path)
        self.__completed_work.append(operation)