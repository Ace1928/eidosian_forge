from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import errno
import fcntl
import hashlib
import io
import os
import pty
import re
import shlex
import subprocess
import time
import typing as t
from functools import wraps
from ansible.errors import (
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.compat import selectors
from ansible.module_utils.six import PY3, text_type, binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import BOOLEANS, boolean
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath, makedirs_safe
@staticmethod
def _sshpass_available() -> bool:
    global SSHPASS_AVAILABLE
    if SSHPASS_AVAILABLE is None:
        try:
            p = subprocess.Popen(['sshpass'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.communicate()
            SSHPASS_AVAILABLE = True
        except OSError:
            SSHPASS_AVAILABLE = False
    return SSHPASS_AVAILABLE