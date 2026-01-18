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
def _change_user_password(self):
    """Change password for SELF.NAME against SELF.PASSWORD.

        Please note that password must be cleartext.
        """
    cmd = self._get_dscl()
    if self.password:
        cmd += ['-passwd', '/Users/%s' % self.name, self.password]
    else:
        cmd += ['-create', '/Users/%s' % self.name, 'Password', '*']
    rc, out, err = self.execute_command(cmd)
    if rc != 0:
        self.module.fail_json(msg='Error when changing password', err=err, out=out, rc=rc)
    return (rc, out, err)