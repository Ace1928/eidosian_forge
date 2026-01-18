from __future__ import (annotations, absolute_import, division, print_function)
import os
import socket
import tempfile
import traceback
import fcntl
import re
import typing as t
from ansible.module_utils.compat.version import LooseVersion
from binascii import hexlify
from ansible.errors import (
from ansible.module_utils.compat.paramiko import PARAMIKO_IMPORT_ERR, paramiko
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def _set_log_channel(self, name: str) -> None:
    """Mimic paramiko.SSHClient.set_log_channel"""
    self._log_channel = name