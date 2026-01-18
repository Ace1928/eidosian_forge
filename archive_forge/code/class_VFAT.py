from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class VFAT(Filesystem):
    INFO = 'fatresize'
    GROW = 'fatresize'
    GROW_MAX_SPACE_FLAGS = ['-s', 'max']

    def __init__(self, module):
        super(VFAT, self).__init__(module)
        if platform.system() == 'FreeBSD':
            self.MKFS = 'newfs_msdos'
        else:
            self.MKFS = 'mkfs.vfat'

    def get_fs_size(self, dev):
        """Get and return size of filesystem, in bytes."""
        cmd = self.module.get_bin_path(self.INFO, required=True)
        dummy, out, dummy = self.module.run_command([cmd, '--info', str(dev)], check_rc=True, environ_update=self.LANG_ENV)
        fssize = None
        for line in out.splitlines()[1:]:
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            param, value = parts
            if param.strip() in ('Size', 'Cur size'):
                fssize = int(value.strip())
                break
        else:
            raise ValueError(repr(out))
        return fssize