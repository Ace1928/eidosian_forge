from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __stage_set_default_subvolume(self, subvolume_path, subvolume_id=None):
    """Add update of the filesystem's default subvolume to the unit of work"""
    self.__unit_of_work.append({'action': self.__SET_DEFAULT_SUBVOLUME_OPERATION, 'target': subvolume_path, 'target_id': subvolume_id})