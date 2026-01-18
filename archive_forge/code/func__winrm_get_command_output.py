from __future__ import (annotations, absolute_import, division, print_function)
import base64
import logging
import os
import re
import traceback
import json
import tempfile
import shlex
import subprocess
import time
import typing as t
import xml.etree.ElementTree as ET
from inspect import getfullargspec
from urllib.parse import urlunsplit
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.plugins.shell.powershell import ShellBase as PowerShellBase
from ansible.utils.hashing import secure_hash
from ansible.utils.display import Display
def _winrm_get_command_output(self, protocol: winrm.Protocol, shell_id: str, command_id: str, try_once: bool=False) -> tuple[bytes, bytes, int]:
    stdout_buffer, stderr_buffer = ([], [])
    command_done = False
    return_code = -1
    while not command_done:
        try:
            stdout, stderr, return_code, command_done = self._winrm_get_raw_command_output(protocol, shell_id, command_id)
            stdout_buffer.append(stdout)
            stderr_buffer.append(stderr)
            try_once = False
        except WinRMOperationTimeoutError:
            if try_once:
                break
            continue
    return (b''.join(stdout_buffer), b''.join(stderr_buffer), return_code)