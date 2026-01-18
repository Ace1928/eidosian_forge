from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __filter_child_subvolumes(self, subvolumes):
    """Filter the provided list of subvolumes to remove any that are a child of another item in the list"""
    filtered = []
    last = None
    ordered = sorted(subvolumes, key=lambda x: x.path)
    for next in ordered:
        if last is None or not next.path[0:len(last)] == last:
            filtered.append(next)
            last = next.path
    return filtered