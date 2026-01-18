from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_create_intermediates(self, closest_subvolume, subvolume_name):
    relative_path = closest_subvolume.get_child_relative_path(self.__name)
    missing_subvolumes = [x for x in relative_path.split(os.path.sep) if len(x) > 0]
    if len(missing_subvolumes) > 1:
        current = closest_subvolume.path
        for s in missing_subvolumes[:-1]:
            separator = os.path.sep if current[-1] != os.path.sep else ''
            current = current + separator + s
            self.__stage_create_subvolume(current, True)