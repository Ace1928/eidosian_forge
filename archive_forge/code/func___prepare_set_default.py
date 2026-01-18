from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_set_default(self):
    subvolume = self.__filesystem.get_subvolume_by_name(self.__name)
    subvolume_id = subvolume.id if subvolume is not None else None
    if self.__filesystem.default_subvolid != subvolume_id:
        self.__stage_set_default_subvolume(self.__name, subvolume_id)