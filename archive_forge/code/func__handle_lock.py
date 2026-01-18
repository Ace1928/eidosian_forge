from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
def _handle_lock(self):
    info = self.user_info()
    if self.password_lock and (not info[1].startswith('*LOCKED*')):
        cmd = [self.module.get_bin_path('pw', True), 'lock', self.name]
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
        return self.execute_command(cmd)
    elif self.password_lock is False and info[1].startswith('*LOCKED*'):
        cmd = [self.module.get_bin_path('pw', True), 'unlock', self.name]
        if self.uid is not None and info[2] != int(self.uid):
            cmd.append('-u')
            cmd.append(self.uid)
        return self.execute_command(cmd)
    return (None, '', '')