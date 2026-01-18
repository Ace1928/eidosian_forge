from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __cleanup_mount(self, mountpoint):
    umount = self.module.get_bin_path('umount', required=True)
    result = self.module.run_command('%s %s' % (umount, mountpoint))
    if result[0] == 0:
        rmdir = self.module.get_bin_path('rmdir', required=True)
        self.module.run_command('%s %s' % (rmdir, mountpoint))