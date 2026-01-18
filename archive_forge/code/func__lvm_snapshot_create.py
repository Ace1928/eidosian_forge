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
def _lvm_snapshot_create(self, source_lv, snapshot_name, snapshot_size_gb=5):
    """Create an LVM snapshot.

        :param source_lv: Name of lv to snapshot
        :type source_lv: ``str``
        :param snapshot_name: Name of lv snapshot
        :type snapshot_name: ``str``
        :param snapshot_size_gb: Size of snapshot to create
        :type snapshot_size_gb: ``int``
        """
    vg = self._get_lxc_vg()
    free_space, measurement = self._get_vg_free_pe(vg_name=vg)
    if free_space < float(snapshot_size_gb):
        message = 'Snapshot size [ %s ] is > greater than [ %s ] on volume group [ %s ]' % (snapshot_size_gb, free_space, vg)
        self.failure(error='Not enough space to create snapshot', rc=2, msg=message)
    build_command = [self.module.get_bin_path('lvcreate', True), '-n', snapshot_name, '-s', os.path.join(vg, source_lv), '-L%sg' % snapshot_size_gb]
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='Failed to Create LVM snapshot %s/%s --> %s' % (vg, source_lv, snapshot_name))