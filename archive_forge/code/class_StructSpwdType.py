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
class StructSpwdType(ctypes.Structure):
    _fields_ = [('sp_namp', ctypes.c_char_p), ('sp_pwdp', ctypes.c_char_p), ('sp_lstchg', ctypes.c_long), ('sp_min', ctypes.c_long), ('sp_max', ctypes.c_long), ('sp_warn', ctypes.c_long), ('sp_inact', ctypes.c_long), ('sp_expire', ctypes.c_long), ('sp_flag', ctypes.c_ulong)]