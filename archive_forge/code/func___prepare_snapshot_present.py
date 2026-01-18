from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_snapshot_present(self):
    source_subvolume = self.__filesystem.get_subvolume_by_name(self.__snapshot_source)
    subvolume = self.__filesystem.get_subvolume_by_name(self.__name)
    subvolume_exists = subvolume is not None
    if subvolume_exists:
        if self.__snapshot_conflict == 'skip':
            return
        elif self.__snapshot_conflict == 'error':
            raise BtrfsModuleException("Target subvolume=%s already exists and snapshot_conflict='error'" % self.__name)
    if source_subvolume is None:
        raise BtrfsModuleException('Source subvolume %s does not exist' % self.__snapshot_source)
    elif subvolume is not None and source_subvolume.id == subvolume.id:
        raise BtrfsModuleException('Snapshot source and target are the same.')
    else:
        self.__stage_required_mount(source_subvolume)
    if subvolume_exists and self.__snapshot_conflict == 'clobber':
        self.__prepare_delete_subvolume_tree(subvolume)
    elif not subvolume_exists:
        self.__prepare_before_create_subvolume(self.__name)
    self.__stage_create_snapshot(source_subvolume, self.__name)