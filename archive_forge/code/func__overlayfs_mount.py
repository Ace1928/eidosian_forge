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
def _overlayfs_mount(self, lowerdir, upperdir, mount_point):
    """mount an lv.

        :param lowerdir: name/path of the lower directory
        :type lowerdir: ``str``
        :param upperdir: name/path of the upper directory
        :type upperdir: ``str``
        :param mount_point: path on the file system that is mounted.
        :type mount_point: ``str``
        """
    build_command = [self.module.get_bin_path('mount', True), '-t', 'overlayfs', '-o', 'lowerdir=%s,upperdir=%s' % (lowerdir, upperdir), 'overlayfs', mount_point]
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='failed to mount overlayfs:%s:%s to %s -- Command: %s' % (lowerdir, upperdir, mount_point, build_command))