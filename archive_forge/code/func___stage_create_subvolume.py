from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __stage_create_subvolume(self, subvolume_path, intermediate=False):
    """
        Add required creation of an intermediate subvolume to the unit of work
        If intermediate is true, the action will be skipped if a directory like file is found at target
        after mounting a parent subvolume
        """
    self.__unit_of_work.append({'action': self.__CREATE_SUBVOLUME_OPERATION, 'target': subvolume_path, 'intermediate': intermediate})