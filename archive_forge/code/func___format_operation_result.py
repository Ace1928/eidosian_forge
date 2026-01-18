from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __format_operation_result(self, operation):
    action_type = operation['action']
    if action_type == self.__CREATE_SUBVOLUME_OPERATION:
        return self.__format_create_subvolume_result(operation)
    elif action_type == self.__CREATE_SNAPSHOT_OPERATION:
        return self.__format_create_snapshot_result(operation)
    elif action_type == self.__DELETE_SUBVOLUME_OPERATION:
        return self.__format_delete_subvolume_result(operation)
    elif action_type == self.__SET_DEFAULT_SUBVOLUME_OPERATION:
        return self.__format_set_default_subvolume_result(operation)
    else:
        raise ValueError("Unknown operation type '%s'" % operation['action'])