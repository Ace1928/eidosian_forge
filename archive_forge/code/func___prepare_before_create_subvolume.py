from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_before_create_subvolume(self, subvolume_name):
    closest_parent = self.__filesystem.get_nearest_subvolume(subvolume_name)
    self.__stage_required_mount(closest_parent)
    if self.__recursive:
        self.__prepare_create_intermediates(closest_parent, subvolume_name)