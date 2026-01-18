from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __mount_subvolume_id_to_tempdir(self, filesystem, subvolid):
    if self.module.check_mode or not self.__automount:
        raise BtrfsModuleException('Unable to temporarily mount required subvolumeswith automount=%s and check_mode=%s' % (self.__automount, self.module.check_mode))
    cache_key = '%s:%d' % (filesystem.uuid, subvolid)
    if cache_key in self.__temporary_mounts:
        return self.__temporary_mounts[cache_key]
    device = filesystem.devices[0]
    mountpoint = tempfile.mkdtemp(dir='/tmp')
    self.__temporary_mounts[cache_key] = mountpoint
    mount = self.module.get_bin_path('mount', required=True)
    command = '%s -o noatime,subvolid=%d %s %s ' % (mount, subvolid, device, mountpoint)
    result = self.module.run_command(command, check_rc=True)
    return mountpoint