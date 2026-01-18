from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class Ext(Filesystem):
    MKFS_FORCE_FLAGS = ['-F']
    MKFS_SET_UUID_OPTIONS = ['-U']
    INFO = 'tune2fs'
    GROW = 'resize2fs'
    CHANGE_UUID = 'tune2fs'
    CHANGE_UUID_OPTION = '-U'

    def get_fs_size(self, dev):
        """Get Block count and Block size and return their product."""
        cmd = self.module.get_bin_path(self.INFO, required=True)
        dummy, out, dummy = self.module.run_command([cmd, '-l', str(dev)], check_rc=True, environ_update=self.LANG_ENV)
        block_count = block_size = None
        for line in out.splitlines():
            if 'Block count:' in line:
                block_count = int(line.split(':')[1].strip())
            elif 'Block size:' in line:
                block_size = int(line.split(':')[1].strip())
            if None not in (block_size, block_count):
                break
        else:
            raise ValueError(repr(out))
        return block_size * block_count