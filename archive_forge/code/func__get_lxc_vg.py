from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _get_lxc_vg(self):
    """Return the name of the Volume Group used in LXC."""
    build_command = [self.module.get_bin_path('lxc-config', True), 'lxc.bdev.lvm.vg']
    rc, vg, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='Failed to read LVM VG from LXC config', command=' '.join(build_command))
    else:
        return str(vg.strip())