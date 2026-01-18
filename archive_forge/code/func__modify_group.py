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
def _modify_group(self):
    """Add or remove SELF.NAME to or from GROUP depending on ACTION.
        ACTION can be 'add' or 'remove' otherwise 'remove' is assumed. """
    rc = 0
    out = ''
    err = ''
    changed = False
    current = set(self._list_user_groups())
    if self.groups is not None:
        target = self.get_groups_set(names_only=True)
    else:
        target = set([])
    if self.append is False:
        for remove in current - target:
            _rc, _out, _err = self.__modify_group(remove, 'delete')
            rc += rc
            out += _out
            err += _err
            changed = True
    for add in target - current:
        _rc, _out, _err = self.__modify_group(add, 'add')
        rc += _rc
        out += _out
        err += _err
        changed = True
    return (rc, out, err, changed)