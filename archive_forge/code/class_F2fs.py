from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class F2fs(Filesystem):
    MKFS = 'mkfs.f2fs'
    INFO = 'dump.f2fs'
    GROW = 'resize.f2fs'

    def __init__(self, module):
        super(F2fs, self).__init__(module)
        mkfs = self.module.get_bin_path(self.MKFS, required=True)
        dummy, out, dummy = self.module.run_command([mkfs, os.devnull], check_rc=False, environ_update=self.LANG_ENV)
        match = re.search('F2FS-tools: mkfs.f2fs Ver: ([0-9.]+) \\(', out)
        if match is not None:
            if LooseVersion(match.group(1)) >= LooseVersion('1.9.0'):
                self.MKFS_FORCE_FLAGS = ['-f']

    def get_fs_size(self, dev):
        """Get sector size and total FS sectors and return their product."""
        cmd = self.module.get_bin_path(self.INFO, required=True)
        dummy, out, dummy = self.module.run_command([cmd, str(dev)], check_rc=True, environ_update=self.LANG_ENV)
        sector_size = sector_count = None
        for line in out.splitlines():
            if 'Info: sector size = ' in line:
                sector_size = int(line.split()[4])
            elif 'Info: total FS sectors = ' in line:
                sector_count = int(line.split()[5])
            if None not in (sector_size, sector_count):
                break
        else:
            raise ValueError(repr(out))
        return sector_size * sector_count