from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __find_default_filesystem(self):
    filesystems = self.__provider.get_filesystems()
    filesystem = None
    if len(filesystems) == 1:
        filesystem = filesystems[0]
    else:
        mounted_filesystems = [x for x in filesystems if x.is_mounted()]
        if len(mounted_filesystems) == 1:
            filesystem = mounted_filesystems[0]
    if filesystem is not None:
        return filesystem
    else:
        raise BtrfsModuleException('Failed to automatically identify targeted filesystem. No explicit device indicated and found %d available filesystems.' % len(filesystems))