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
def _user_attr_info(self):
    info = [''] * 3
    with open(self.USER_ATTR, 'r') as file_handler:
        for line in file_handler:
            lines = line.strip().split('::::')
            if lines[0] == self.name:
                tmp = dict((x.split('=') for x in lines[1].split(';')))
                info[0] = tmp.get('profiles', '')
                info[1] = tmp.get('auths', '')
                info[2] = tmp.get('roles', '')
    return info