from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __stage_delete_subvolume(self, subvolume):
    """Add deletion of the target subvolume to the unit of work"""
    self.__unit_of_work.append({'action': self.__DELETE_SUBVOLUME_OPERATION, 'target': subvolume.path, 'target_id': subvolume.id})