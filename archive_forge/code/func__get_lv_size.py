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
def _get_lv_size(self, lv_name):
    """Return the available size of a given LV.

        :param lv_name: Name of volume.
        :type lv_name: ``str``
        :returns: size and measurement of an LV
        :type: ``tuple``
        """
    vg = self._get_lxc_vg()
    lv = os.path.join(vg, lv_name)
    build_command = ['lvdisplay', lv, '--units', 'g']
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='failed to read lv %s' % lv, command=' '.join(build_command))
    lv_info = [i.strip() for i in stdout.splitlines()][1:]
    _free_pe = [i for i in lv_info if i.startswith('LV Size')]
    free_pe = _free_pe[0].split()
    return (self._roundup(float(free_pe[-2])), free_pe[-1])