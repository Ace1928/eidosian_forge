from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __execute_unit_of_work(self):
    self.__check_required_mounts()
    for op in self.__unit_of_work:
        if op['action'] == self.__CREATE_SUBVOLUME_OPERATION:
            self.__execute_create_subvolume(op)
        elif op['action'] == self.__CREATE_SNAPSHOT_OPERATION:
            self.__execute_create_snapshot(op)
        elif op['action'] == self.__DELETE_SUBVOLUME_OPERATION:
            self.__execute_delete_subvolume(op)
        elif op['action'] == self.__SET_DEFAULT_SUBVOLUME_OPERATION:
            self.__execute_set_default_subvolume(op)
        else:
            raise ValueError("Unknown operation type '%s'" % op['action'])