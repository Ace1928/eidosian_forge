from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __format_set_default_subvolume_result(self, operation):
    target = operation['target']
    if 'target_id' in operation:
        target_id = operation['target_id']
    else:
        target_subvolume = self.__filesystem.get_subvolume_by_name(target)
        target_id = target_subvolume.id if target_subvolume is not None else self.__UNKNOWN_SUBVOLUME_ID
    return "Updated default subvolume to '%s' (%s)" % (target, target_id)