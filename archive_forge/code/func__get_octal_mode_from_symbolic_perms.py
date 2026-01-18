from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
@staticmethod
def _get_octal_mode_from_symbolic_perms(path_stat, user, perms, use_umask, prev_mode=None):
    if prev_mode is None:
        prev_mode = stat.S_IMODE(path_stat.st_mode)
    is_directory = stat.S_ISDIR(path_stat.st_mode)
    has_x_permissions = prev_mode & EXEC_PERM_BITS > 0
    apply_X_permission = is_directory or has_x_permissions
    umask = os.umask(0)
    os.umask(umask)
    rev_umask = umask ^ PERM_BITS
    if apply_X_permission:
        X_perms = {'u': {'X': stat.S_IXUSR}, 'g': {'X': stat.S_IXGRP}, 'o': {'X': stat.S_IXOTH}}
    else:
        X_perms = {'u': {'X': 0}, 'g': {'X': 0}, 'o': {'X': 0}}
    user_perms_to_modes = {'u': {'r': rev_umask & stat.S_IRUSR if use_umask else stat.S_IRUSR, 'w': rev_umask & stat.S_IWUSR if use_umask else stat.S_IWUSR, 'x': rev_umask & stat.S_IXUSR if use_umask else stat.S_IXUSR, 's': stat.S_ISUID, 't': 0, 'u': prev_mode & stat.S_IRWXU, 'g': (prev_mode & stat.S_IRWXG) << 3, 'o': (prev_mode & stat.S_IRWXO) << 6}, 'g': {'r': rev_umask & stat.S_IRGRP if use_umask else stat.S_IRGRP, 'w': rev_umask & stat.S_IWGRP if use_umask else stat.S_IWGRP, 'x': rev_umask & stat.S_IXGRP if use_umask else stat.S_IXGRP, 's': stat.S_ISGID, 't': 0, 'u': (prev_mode & stat.S_IRWXU) >> 3, 'g': prev_mode & stat.S_IRWXG, 'o': (prev_mode & stat.S_IRWXO) << 3}, 'o': {'r': rev_umask & stat.S_IROTH if use_umask else stat.S_IROTH, 'w': rev_umask & stat.S_IWOTH if use_umask else stat.S_IWOTH, 'x': rev_umask & stat.S_IXOTH if use_umask else stat.S_IXOTH, 's': 0, 't': stat.S_ISVTX, 'u': (prev_mode & stat.S_IRWXU) >> 6, 'g': (prev_mode & stat.S_IRWXG) >> 3, 'o': prev_mode & stat.S_IRWXO}}
    for key, value in X_perms.items():
        user_perms_to_modes[key].update(value)

    def or_reduce(mode, perm):
        return mode | user_perms_to_modes[user][perm]
    return reduce(or_reduce, perms, 0)