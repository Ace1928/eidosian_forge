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
def _lvm_lv_list(self):
    """Return a list of all lv in a current vg."""
    vg = self._get_lxc_vg()
    build_command = [self.module.get_bin_path('lvs', True)]
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='Failed to get list of LVs', command=' '.join(build_command))
    all_lvms = [i.split() for i in stdout.splitlines()][1:]
    return [lv_entry[0] for lv_entry in all_lvms if lv_entry[1] == vg]